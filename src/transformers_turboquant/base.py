"""Abstract base class for TurboQuant attention replacements."""

from abc import ABC, abstractmethod
from typing import ClassVar

import torch.nn as nn

from turboquant_pytorch.compressors import (
    TurboQuantCompressorMSE,
    TurboQuantCompressorV2,
)


class TurboQuantAttentionBase(ABC, nn.Module):
    """
    Base class for all TurboQuant attention replacements.

    Each family subclass declares which HuggingFace attention class names it
    can replace via ATTENTION_CLASS_NAMES, and implements from_original() to
    copy weights from the original module and forward() to run compressed attention.
    """

    ATTENTION_CLASS_NAMES: ClassVar[tuple[str, ...]]

    @classmethod
    @abstractmethod
    def from_original(
        cls,
        original: nn.Module,
        key_compressor: TurboQuantCompressorV2,
        val_compressor: TurboQuantCompressorMSE,
    ) -> "TurboQuantAttentionBase":
        """Copy weights from original attention module and store compressors."""
        ...

    @abstractmethod
    def forward(self, hidden_states, **kwargs):
        """Run compressed attention forward pass."""
        ...
