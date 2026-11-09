from torch import nn
import torch


class PosEmbeddingLayer(nn.Module):
    """Learned absolute position embedding (base, non-flex version; FlexGPT)."""

    def __init__(self, seq_length, hidden_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(
            1, seq_length, hidden_dim).normal_(std=0.02))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Same as flex_modules.PosEmbeddingLayer at a fixed width.
        return x + self.embedding[:, :x.shape[1]]


class LinearHead(nn.Linear):
    """
    This class only exists for compatibility with the level delta system, as we
    need a distinct type for a linear select and a shared weights linear layer.
    """

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
