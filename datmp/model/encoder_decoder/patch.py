import math

import torch
import torch.nn as nn
from transformers.activations import ACT2FN


class PatchEmbedding(nn.Module):
    """
    Patch Embedding module using convolution.
    Converts input vectors into patches using 1D convolution.
    linear -> activation -> dropout -> conv1d -> dropout
    """

    def __init__(
        self,
        seq_len: int,
        hidden_size: int = 64,
        patch_size: int = 40,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            seq_len: Sequence length
            hidden_size: Hidden size for convolution
            patch_size: Size of each patch
            dropout: Dropout rate
            activation: Activation function name
        """
        super().__init__()

        assert seq_len % patch_size == 0, f"Sequence length {seq_len} must be divisible by patch size {patch_size}"
        num_patches = seq_len // patch_size

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # CLS tokens for task and domain
        self.cls_token = nn.ParameterDict(
            {
                "task": nn.Parameter(torch.randn(1, 1, hidden_size)),
                "domain": nn.Parameter(torch.randn(1, 1, hidden_size)),
            }
        )

        # Positional encoding
        self.register_buffer(
            "pos_encoding",
            self._generate_sinusoidal_positional_encoding(num_patches + 2, hidden_size),
            persistent=False,
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len)

        Returns:
            Output tensor of shape (batch_size, num_patches + 2, hidden_size)
            [domain_cls_token, task_cls_token, ...patches...]
        """
        b = x.shape[0]

        x = x.unsqueeze(1)  # (batch_size, 1, seq_len)
        x = self.conv(x)  # (batch_size, hidden_size, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, hidden_size)

        # Add CLS tokens
        domain_cls_token = self.cls_token["domain"].expand(b, -1, -1)  # (batch_size, 1, hidden_size)
        task_cls_token = self.cls_token["task"].expand(b, -1, -1)  # (batch_size, 1, hidden_size)

        # Concatenate: [domain_cls, task_cls, ...patches...]
        x = torch.cat([domain_cls_token, task_cls_token, x], dim=1)  # (batch_size, num_patches + 2, hidden_size)

        # Add positional encoding
        pos_encoding = self.pos_encoding.to(x.device)
        x = x + pos_encoding

        x = self.layer_norm(x)
        x = self.dropout(x)

        return x

    def _generate_sinusoidal_positional_encoding(self, max_length, dim):
        """
        Generate sinusoidal positional encoding.

        Args:
            max_length: Maximum sequence length
            dim: Feature dimension

        Returns:
            Sinusoidal positional encoding (Tensor) of shape (1, seq_len, dim)
        """
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        pos_encoding = torch.zeros((max_length, dim))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)  # (1, seq_len, dim)


class PatchTransformerEncoder(nn.Module):
    """
    Patch-based Transformer Encoder.
    Converts input vectors into patches and processes them with Transformer.
    """

    def __init__(
        self,
        seq_len: int,
        hidden_size: int = 64,
        patch_size: int = 40,
        dropout: float = 0.1,
        activation: str = "gelu",
        num_heads: int = 4,
        num_layers: int = 6,
    ) -> None:
        """
        Args:
            seq_len: Sequence length
            hidden_size: Hidden size for transformer (should match embedding_dim)
            patch_size: Size of each patch
            dropout: Dropout rate
            activation: Activation function name
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
        """
        super().__init__()

        # Patch embedding layer using convolution
        self.patch_embedding = PatchEmbedding(
            seq_len=seq_len,
            hidden_size=hidden_size,
            patch_size=patch_size,
            dropout=dropout,
        )

        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True,
                activation=activation,
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size),
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len)

        Returns:
            Output tensor of shape (batch_size, num_patches + 2, hidden_size)
            [domain_cls_token, task_cls_token, ...patches...]
        """
        # Convert to patches
        x = self.patch_embedding(x)  # (batch_size, num_patches + 2, hidden_size)
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch_size, num_patches + 2, hidden_size)

        return x


class PatchDecoder(nn.Module):
    """
    Patch-based Decoder for reconstructing original input from encoded features.
    Uses transposed convolution to reconstruct patches back to original dimension.
    """

    def __init__(
        self,
        seq_len: int = 512,
        hidden_size: int = 64,
        patch_size: int = 40,
        activation: str = "gelu",
        dropout: float = 0.1,
    ):
        """
        Args:
            seq_len: Sequence length
            hidden_size: Hidden size from encoder (output channels of conv1d)
            patch_size: Size of each patch
            activation: Activation function name
            dropout: Dropout rate
        """
        super().__init__()

        # Transposed convolution to reconstruct patches
        self.unconv = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=hidden_size,
                out_channels=1,
                kernel_size=patch_size,
                stride=patch_size,
            ),
        )
        self.layer_norm = nn.LayerNorm(seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for reconstruction.

        Args:
            x: Encoded features of shape (batch_size, num_patches, hidden_size)

        Returns:
            Reconstructed tensor of shape (batch_size, seq_len)
        """
        # Transpose and reconstruct using transposed convolution
        x = x.transpose(1, 2)  # (batch_size, num_patches, hidden_size)
        x = self.unconv(x)  # (batch_size, 1, seq_len)
        x = x.squeeze(1)  # (batch_size, seq_len)
        return x
