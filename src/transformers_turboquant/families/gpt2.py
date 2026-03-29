"""TurboQuant attention replacement for GPT-2 family models."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers_turboquant.base import TurboQuantAttentionBase
from transformers_turboquant.registry import register_family
from turboquant_pytorch.compressors import (
    TurboQuantCompressorMSE,
    TurboQuantCompressorV2,
)


@register_family("gpt2")
class TurboQuantGPT2Attention(TurboQuantAttentionBase):
    """
    Drop-in replacement for GPT2Attention that uses TurboQuant compressed KV.

    Copies all projections from the original module. The forward recomputes
    full-sequence attention on every call (no incremental KV cache), which is
    correct for both training and inference.
    """

    ATTENTION_CLASS_NAMES = ("GPT2Attention",)

    @classmethod
    def from_original(
        cls,
        original: nn.Module,
        key_compressor: TurboQuantCompressorV2,
        val_compressor: TurboQuantCompressorMSE,
    ) -> "TurboQuantGPT2Attention":
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        # Copy projection layers (Conv1D — register as submodules)
        obj.c_attn = original.c_attn
        obj.c_proj = original.c_proj
        obj.attn_dropout = original.attn_dropout
        obj.resid_dropout = original.resid_dropout
        # Scalars
        obj.embed_dim = original.embed_dim
        obj.num_heads = original.num_heads
        obj.head_dim = original.head_dim
        # Compressors (plain Python objects, not nn.Modules)
        obj.key_compressor = key_compressor
        obj.val_compressor = val_compressor
        return obj

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        B, T, _ = hidden_states.shape

        # Project Q, K, V from combined c_attn weight
        qkv = self.c_attn(hidden_states)  # (B, T, 3 * embed_dim)
        q, k, v = qkv.split(self.embed_dim, dim=2)

        # Reshape to (B, H, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Compress K and V — no_grad is enforced by compressor decorators
        compressed_k = self.key_compressor.compress(k)
        compressed_v = self.val_compressor.compress(v)

        # Asymmetric attention scores: <Q, K> estimated from compressed K
        scores = self.key_compressor.asymmetric_attention_scores(
            q, compressed_k
        )  # (B, H, T, T)
        scores = scores / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # additive causal mask (−inf for future tokens)
            scores = scores + attention_mask

        weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
        weights = self.attn_dropout(weights)

        # Reconstruct V and compute weighted sum
        values_recon = self.val_compressor.decompress(compressed_v).to(
            q.dtype
        )  # (B, H, T, head_dim)
        attn_output = torch.matmul(weights, values_recon)  # (B, H, T, head_dim)

        # Merge heads and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, T, self.embed_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        attn_weights = weights if output_attentions else None
        return attn_output, attn_weights
