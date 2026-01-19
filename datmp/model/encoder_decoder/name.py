import math

import torch
import torch.nn as nn
from transformers.activations import ACT2FN


class NameEmbedding(nn.Module):
    """
    Name embedding layer for input_ids.
    Converts name IDs to dense embeddings.
    """

    def __init__(
        self,
        seq_len: int,
        hidden_size: int = 64,
        dropout: float = 0.1,
    ):
        """
        Args:
            seq_len: Sequence length
            embedding_dim: Embedding dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.name_embedding = nn.Embedding(seq_len, hidden_size)

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
            self._generate_sinusoidal_positional_encoding(seq_len + 2, hidden_size),
            persistent=False,
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Name IDs of shape (batch_size, seq_len)

        Returns:
            Embedded names of shape (batch_size, seq_len + 2, hidden_size)
            [domain_cls_token, task_cls_token, ...name_embeddings...]
        """
        b = input_ids.shape[0]

        # Name embeddings
        x = self.name_embedding(input_ids)  # (batch_size, seq_len, hidden_size)

        # Add CLS tokens
        domain_cls_token = self.cls_token["domain"].expand(b, -1, -1)  # (batch_size, 1, hidden_size)
        task_cls_token = self.cls_token["task"].expand(b, -1, -1)  # (batch_size, 1, hidden_size)

        # Concatenate: [domain_cls, task_cls, ...name_embeddings...]
        x = torch.cat([domain_cls_token, task_cls_token, x], dim=1)  # (batch_size, seq_len + 2, hidden_size)

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
            seq_len: Sequence length
            dim: Feature dimension

        Returns:
            Sinusoidal positional encoding (Tensor)
        """
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        pos_encoding = torch.zeros((max_length, dim))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)  # (1, seq_len, dim)


class NameTransformerEncoder(nn.Module):
    """
    Transformer Encoder for name-encoded data (input_ids format).
    Processes name sequences using embeddings and transformer layers.
    """

    def __init__(
        self,
        seq_len: int,
        hidden_size: int = 64,
        dropout: float = 0.1,
        activation: str = "gelu",
        num_heads: int = 4,
        num_layers: int = 6,
    ) -> None:
        """
        Args:
            seq_len: Sequence length
            hidden_size: Hidden size for transformer
            dropout: Dropout rate
            activation: Activation function name
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
        """
        super().__init__()

        # Token embedding layer
        self.name_embedding = NameEmbedding(
            seq_len=seq_len,
            hidden_size=hidden_size,
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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Name IDs of shape (batch_size, seq_len)

        Returns:
            Output tensor of shape (batch_size, seq_len + 2, hidden_size)
            [domain_cls_token, task_cls_token, ...name_embeddings...]
        """

        # Name embeddings
        x = self.name_embedding(input_ids)  # (batch_size, seq_len + 2, hidden_size)
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch_size, seq_len + 2, hidden_size)

        return x


class NameDecoder(nn.Module):
    """
    Decoder for reconstructing name sequences from encoded features.
    """

    def __init__(
        self,
        seq_len: int = 512,
        encoder_hidden_size: int = 64,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        Args:
            encoder_hidden_size: Hidden size from encoder
            seq_len: Sequence length
            dropout: Dropout rate
            activation: Activation function name
        """
        super().__init__()

        self.hidden_size = encoder_hidden_size
        self.seq_len = seq_len

        # Projection layers
        self.proj = nn.Linear(encoder_hidden_size, seq_len)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for reconstruction.

        Args:
            features: Combined features of shape (batch_size, seq_len, encoder_hidden_size)
                Note: This should be private[:, 2:, :] + shared[:, 2:, :], combined before calling decoder

        Returns:
            Reconstructed features of shape (batch_size, seq_len)
        """
        # Project
        x = self.proj(features)  # (batch_size, seq_len)

        return x
