"""
Slow model-based tests: needle-in-haystack attention score comparison.

Loads Qwen2.5-3B-Instruct (4-bit BitsAndBytes) and runs a fixed needle-in-haystack
prompt through three modes:
  1. Baseline  — original model, no TurboQuant
  2. TurboQuant PyTorch — use_triton=False
  3. TurboQuant Triton  — use_triton=True

Collects attention score variance per (layer, head) at b ∈ {2, 3, 4} and prints
a rich table.  The test passes regardless of the numbers.
"""

from __future__ import annotations


def _model_available() -> bool:
    """Return True if CUDA is available and required packages can be imported."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        import transformers  # noqa: F401
        import bitsandbytes  # noqa: F401

        return True
    except ImportError:
        return False


def _rich_available() -> bool:
    try:
        import rich  # noqa: F401

        return True
    except ImportError:
        return False


import pytest

# ---------------------------------------------------------------------------
# Constants (same as validate.py)
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
NEEDLE = "The secret project code name is AURORA-7749."
QUESTION = "What is the secret project code name?"
FILLER = (
    "The quarterly financial review meeting covered several topics including\n"
    "budget allocations for the upcoming fiscal year, departmental spending reports, and projected\n"
    "revenue streams from various business units. The committee discussed infrastructure upgrades\n"
    "planned for the western regional offices and noted that maintenance schedules should be\n"
    "coordinated with the facilities management team. Several action items were assigned to team\n"
    "leads for follow-up before the next meeting cycle.\n\n"
)


def _build_prompt(tokenizer, target_tokens: int = 2048, needle_pos: float = 0.5) -> str:
    filler_len = len(tokenizer.encode(FILLER))
    n_reps = max(1, target_tokens // filler_len)
    needle_idx = int(n_reps * needle_pos)
    parts: list[str] = []
    for i in range(n_reps):
        if i == needle_idx:
            parts.append(f"\n--- Memo ---\n{NEEDLE}\n--- End ---\n\n")
        parts.append(FILLER)
    haystack = "".join(parts)
    return (
        f"<|im_start|>user\n{haystack}\nQuestion: {QUESTION}"
        "<|im_end|>\n<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attn_variance(scores: "torch.Tensor") -> "torch.Tensor":
    """
    Compute variance of attention scores over the S_k (key-sequence) dimension.

    Args:
        scores: (B, H, S_q, S_k)  — raw (pre-softmax) dot-product scores

    Returns:
        var: (H,)  — mean variance across batch and query positions
    """
    import torch

    # scores: (B, H, S_q, S_k)
    var_sq = scores.float().var(dim=-1)  # (B, H, S_q)
    return var_sq.mean(dim=(0, 2))       # (H,)


def _run_mode(
    cache: object,
    head_dim: int,
    n_layers: int,
    bits: int,
    device: str,
    use_triton: bool | None,
) -> "dict[tuple[int,int], float]":
    """
    Compute per-(layer, head) attention score variance for one mode.

    Args:
        cache:      HuggingFace past_key_values
        head_dim:   size of each key/value head vector
        n_layers:   number of transformer layers
        bits:       quantisation bit-width (2, 3, or 4)
        device:     torch device string, e.g. "cuda"
        use_triton: None → baseline (no TurboQuant), else passed to compressor

    Returns:
        dict mapping (layer_idx, head_idx) -> mean variance (scalar float)
    """
    import torch
    import sys
    import os

    # Make sure the turboquant package is importable when running from tests/
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from turboquant_pytorch.compressors import TurboQuantCompressorV2

    result: dict[tuple[int, int], float] = {}

    for layer_idx in range(n_layers):
        keys = cache.layers[layer_idx].keys    # (B, H, S, D)
        B, H, S, D = keys.shape

        # Query = last token's key vector (proxy for generation-time query)
        query = keys[:, :, -1:, :]  # (B, H, 1, D)

        if use_triton is None:
            # Baseline: exact dot-product scores
            scores = torch.matmul(
                query.float(), keys.float().transpose(-2, -1)
            )  # (B, H, 1, S)
        else:
            comp = TurboQuantCompressorV2(
                D,
                bits,
                seed=layer_idx * 1000,
                device=device,
                use_triton=use_triton,
            )
            compressed_k = comp.compress(keys)
            scores = comp.asymmetric_attention_scores(query, compressed_k)  # (B, H, 1, S)

        var_per_head = _attn_variance(scores)  # (H,)
        for h in range(H):
            result[(layer_idx, h)] = var_per_head[h].item()

    return result


# ---------------------------------------------------------------------------
# The slow test
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(
    not _model_available(),
    reason="Qwen2.5-3B-Instruct not available or no GPU",
)
def test_needle_in_haystack_attention_variance() -> None:
    """
    Load Qwen2.5-3B-Instruct, run a needle-in-haystack prompt, and compare
    attention score variances across baseline / TurboQuant-PyTorch / TurboQuant-Triton
    for bits ∈ {2, 3, 4}.

    The test is purely a reporting test — it always passes.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device = "cuda"

    # ------------------------------------------------------------------
    # 1. Load model & tokenizer
    # ------------------------------------------------------------------
    print("\nLoading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        ),
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    print(
        f"Loaded. GPU: {torch.cuda.memory_allocated() // 1024 // 1024} MB",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 2. Build prompt and tokenize
    # ------------------------------------------------------------------
    TARGET_TOKENS = 2048
    prompt = _build_prompt(tokenizer, target_tokens=TARGET_TOKENS)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=TARGET_TOKENS + 256,
    ).to(device)

    print(
        f"Prompt tokens: {inputs['input_ids'].shape[1]}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 3. Single forward pass to capture KV cache
    # ------------------------------------------------------------------
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_attentions=False)

    cache = outputs.past_key_values
    n_layers: int = len(cache.layers)
    head_dim: int = cache.layers[0].keys.shape[-1]
    n_heads: int = cache.layers[0].keys.shape[1]

    print(
        f"Layers: {n_layers}, heads: {n_heads}, head_dim: {head_dim}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 4. Collect variances for all modes and bit-widths
    # ------------------------------------------------------------------
    # Structure: results[(bits, layer, head)] = {"baseline": v, "pytorch": v, "triton": v}
    # We'll iterate bits and modes, storing variance dicts.

    # Pre-allocate per-(layer, head) storage: list of dicts keyed by bits
    # rows: list of (layer, head, bits, baseline_var, pytorch_var, triton_var, pt_delta, triton_delta)
    rows: list[tuple] = []

    for bits in [2, 3, 4]:
        print(f"\nBits = {bits}:", flush=True)

        baseline_vars = _run_mode(cache, head_dim, n_layers, bits, device, use_triton=None)
        print("  baseline done", flush=True)

        pytorch_vars = _run_mode(cache, head_dim, n_layers, bits, device, use_triton=False)
        print("  pytorch done", flush=True)

        triton_vars = _run_mode(cache, head_dim, n_layers, bits, device, use_triton=True)
        print("  triton done", flush=True)

        for (layer_idx, head_idx), bv in baseline_vars.items():
            pv = pytorch_vars[(layer_idx, head_idx)]
            tv = triton_vars[(layer_idx, head_idx)]
            rows.append(
                (
                    layer_idx,
                    head_idx,
                    bits,
                    bv,
                    pv,
                    tv,
                    abs(pv - bv),
                    abs(tv - bv),
                )
            )

    # ------------------------------------------------------------------
    # 5. Print rich table (or plain fallback)
    # ------------------------------------------------------------------
    if _rich_available():
        from rich.console import Console
        from rich.table import Table

        table = Table(
            title="TurboQuant Attention Variance (needle-in-haystack)",
            show_lines=False,
        )
        table.add_column("layer", justify="right", style="cyan")
        table.add_column("head", justify="right", style="cyan")
        table.add_column("bits", justify="right", style="magenta")
        table.add_column("baseline_var", justify="right")
        table.add_column("pytorch_var", justify="right")
        table.add_column("triton_var", justify="right")
        table.add_column("pytorch_vs_baseline_Δ", justify="right", style="yellow")
        table.add_column("triton_vs_baseline_Δ", justify="right", style="yellow")

        for layer_idx, head_idx, bits, bv, pv, tv, pt_delta, triton_delta in rows:
            table.add_row(
                str(layer_idx),
                str(head_idx),
                str(bits),
                f"{bv:.6f}",
                f"{pv:.6f}",
                f"{tv:.6f}",
                f"{pt_delta:.6f}",
                f"{triton_delta:.6f}",
            )

        console = Console()
        console.print(table)
    else:
        # Plain-text fallback
        header = (
            f"{'layer':>5}  {'head':>5}  {'bits':>4}  "
            f"{'baseline_var':>12}  {'pytorch_var':>11}  {'triton_var':>10}  "
            f"{'pytorch_vs_baseline_Δ':>21}  {'triton_vs_baseline_Δ':>20}"
        )
        print(header)
        print("-" * len(header))
        for layer_idx, head_idx, bits, bv, pv, tv, pt_delta, triton_delta in rows:
            print(
                f"{layer_idx:>5}  {head_idx:>5}  {bits:>4}  "
                f"{bv:>12.6f}  {pv:>11.6f}  {tv:>10.6f}  "
                f"{pt_delta:>21.6f}  {triton_delta:>20.6f}"
            )

    # Test always passes — this is a reporting test, not a correctness test.
    assert True
