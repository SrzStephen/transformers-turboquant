"""TurboQuant attention replacement for LLaMA and Mistral family models."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from transformers_turboquant.base import TurboQuantAttentionBase
from transformers_turboquant.registry import register_family
from turboquant_pytorch.compressors import (
    TurboQuantCompressorMSE,
    TurboQuantCompressorV2,
)


@register_family("llama", "mistral")
class TurboQuantLlamaAttention(TurboQuantAttentionBase):
    """
    Drop-in replacement for LlamaAttention, LlamaSdpaAttention,
    MistralAttention, and MistralSdpaAttention.

    Architecture identical to Qwen2 (GQA + pre-computed RoPE).
    """

    ATTENTION_CLASS_NAMES = (
        "LlamaAttention",
        "LlamaSdpaAttention",
        "LlamaFlashAttention2",
        "MistralAttention",
        "MistralSdpaAttention",
        "MistralFlashAttention2",
    )

    @classmethod
    def from_original(
        cls,
        original: nn.Module,
        key_compressor: TurboQuantCompressorV2,
        val_compressor: TurboQuantCompressorMSE,
    ) -> "TurboQuantLlamaAttention":
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.q_proj = original.q_proj
        obj.k_proj = original.k_proj
        obj.v_proj = original.v_proj
        obj.o_proj = original.o_proj
        obj.num_heads = original.num_heads
        obj.num_key_value_heads = original.num_key_value_heads
        obj.num_key_value_groups = original.num_heads // original.num_key_value_heads
        obj.head_dim = original.head_dim
        obj.hidden_size = original.hidden_size
        obj.key_compressor = key_compressor
        obj.val_compressor = val_compressor
        return obj

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        for comp in (self.key_compressor, self.val_compressor):
            comp.Pi = fn(comp.Pi)
            comp.PiT = fn(comp.PiT)
            comp.S = fn(comp.S)
            comp.centroids = fn(comp.centroids)
            comp.boundaries = fn(comp.boundaries)
            comp.device = str(comp.Pi.device)
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_value=None,
        cache_position: torch.LongTensor | None = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple | None]:
        B, T, _ = hidden_states.shape

        q = (
            self.q_proj(hidden_states)
            .view(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(B, T, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(B, T, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        compressed_k = self.key_compressor.compress(k)
        compressed_v = self.val_compressor.compress(v)

        scores = self.key_compressor.asymmetric_attention_scores(q, compressed_k)
        scores = scores / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores + attention_mask

        weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
        values_recon = self.val_compressor.decompress(compressed_v).to(q.dtype)
        attn_output = torch.matmul(weights, values_recon)

        attn_output = attn_output.transpose(1, 2).reshape(
            B, T, self.num_heads * self.head_dim
        )
        attn_output = self.o_proj(attn_output)

        attn_weights = weights if output_attentions else None
        return attn_output, attn_weights, None
