# dual encoder model for scgpt
import math

import torch
import torch.nn as nn


class Lambda(nn.Module):
    """Lambda layer to wrap arbitrary functions as nn.Module."""

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.func(x)


class DualTransformerEmbedding(nn.Module):
    """
    Unified embedding layer for Dual Transformer.
    Simultaneously encodes gene names (token IDs) and gene expressions.
    Supports both continuous and discrete (binned) expression values.
    """

    def __init__(
        self,
        seq_len: int,
        hidden_size: int = 64,
        dropout: float = 0.1,
        binning: int = None,
    ):
        """
        Args:
                seq_len: Sequence length for gene name token embedding
                hidden_size: Output embedding dimension
                seq_len: Sequence length for positional encoding
                dropout: Dropout rate
                binning: Number of bins for discrete expression values.
                                  If None, uses continuous values (Linear projection).
                                  If provided, uses discrete values (Embedding layer).
        """
        super().__init__()
        self.seq_len = seq_len
        self.binning = binning
        self.is_discrete_expr = binning is not None
        self.hidden_size = hidden_size

        # Gene name embedding (token IDs)
        self.name_embedding = nn.Embedding(seq_len, hidden_size)

        # Gene expression embedding
        if self.is_discrete_expr:
            # Discrete expression values (after binning) - use Embedding
            # binning + 1 to account for bin indices [0, binning]
            self.expr_embedding = nn.Embedding(binning, hidden_size)
        else:
            # Continuous expression values - use Linear projection
            # self.expr_embedding = nn.Linear(1, hidden_size)
            self.expr_embedding = nn.Sequential(
                Lambda(lambda x: x.unsqueeze(-1)),  # (batch_size, seq_len) -> (batch_size, seq_len, 1)
                nn.Linear(1, hidden_size),  # (batch_size, seq_len, 1) -> (batch_size, seq_len, hidden_size)
            )

        self.pos_norm_name = nn.LayerNorm(hidden_size)
        self.pos_norm_expr = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "pos_encoding",
            self._generate_sinusoidal_positional_encoding(seq_len, hidden_size),
            persistent=False,
        )

    def forward(self, name, expr):
        """
        Forward pass.

        Args:
                name: Gene name token IDs [batch_size, seq_len]
                expr: Gene expression values
                        - If discrete (binning provided): [batch_size, seq_len] (integer bin indices)
                        - If continuous (binning=None): [batch_size, seq_len] or [batch_size, seq_len, 1] (float values)

        Returns:
                name_emb: Embedded gene names [batch_size, seq_len, hidden_size]
                expr_emb: Embedded gene expressions [batch_size, seq_len, hidden_size]
        """

        # Ensure tensors are on the same device
        name_emb = self.name_embedding(name)  # [batch_size, seq_len, hidden_size]

        # Encode gene expressions
        expr_emb = self.expr_embedding(expr)  # [batch_size, seq_len, hidden_size]

        # Apply positional encoding, and dropout to both embeddings
        pos_encoding = self.pos_encoding.to(name_emb.device)

        name_emb = self.pos_norm_name(name_emb + pos_encoding)
        expr_emb = self.pos_norm_expr(expr_emb + pos_encoding)

        name_emb = self.dropout(name_emb)
        expr_emb = self.dropout(expr_emb)

        return name_emb, expr_emb

    def _generate_sinusoidal_positional_encoding(self, seq_len, dim):
        """
        Generate sinusoidal positional encoding.

        Args:
                seq_len: Sequence length
                dim: Feature dimension

        Returns:
                Sinusoidal positional encoding (Tensor) of shape (1, seq_len, dim)
        """
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        pos_encoding = torch.zeros((seq_len, dim))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)  # (1, seq_len, dim)


class DualTransformerEncoder(nn.Module):
    """
    Dual Cross-Attention Encoder for Dual Transformer.

    This encoder processes gene names and expressions using dual cross-attention layers,
    where name sequences attend to expression sequences and vice versa.
    Integrates DualTransformerEmbedding for encoding gene names and expressions.
    """

    def __init__(
        self,
        seq_len: int,
        hidden_size: int = 64,
        num_heads_cross_attn: int = 4,
        num_layers_cross_attn: int = 6,
        num_heads_encoder: int = 4,
        num_layers_encoder: int = 6,
        dropout: float = 0.1,
        activation: str = "gelu",
        binning: int = None,
    ):
        """
        Args:
                seq_len: Sequence length for gene name token embedding
                hidden_size: Hidden dimension size
                num_heads: Number of attention heads
                num_layers: Number of encoder layers
                dropout: Dropout rate
                activation: Activation function name
                binning: Number of bins for discrete expression values. If None, uses continuous values.
        """
        super().__init__()

        # Embedding layer for gene names and expressions
        self.embedding = DualTransformerEmbedding(
            seq_len=seq_len,
            hidden_size=hidden_size,
            dropout=dropout,
            binning=binning,
        )

        self.name_cross_attn_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads_cross_attn,
                    dim_feedforward=hidden_size * 4,
                    dropout=dropout,
                    batch_first=True,
                    activation=activation,
                )
                for _ in range(num_layers_cross_attn)
            ]
        )

        self.expr_cross_attn_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads_cross_attn,
                    dim_feedforward=hidden_size * 4,
                    dropout=dropout,
                    batch_first=True,
                    activation=activation,
                )
                for _ in range(num_layers_cross_attn)
            ]
        )

        # CLS tokens for task and domain
        self.cls_token = nn.ParameterDict(
            {
                "task": nn.Parameter(torch.randn(1, 1, hidden_size)),
                "domain": nn.Parameter(torch.randn(1, 1, hidden_size)),
            }
        )

        # Transformer Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads_encoder,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True,
                activation=activation,
            ),
            num_layers=num_layers_encoder,
            norm=nn.LayerNorm(hidden_size),
        )

    def forward(self, name: torch.Tensor, expr: torch.Tensor):
        """
        Forward pass of dual cross-attention encoder.

        Args:
            name: Gene name token IDs [batch_size, seq_len]
            expr: Gene expression values [batch_size, seq_len]

        Returns:
            concatenated_output: Concatenated output with CLS token [batch_size, 1 + seq_len + seq_len, hidden_size]
            encoded_name: Processed name sequence [batch_size, seq_len, hidden_size]
            encoded_expr: Processed expression sequence [batch_size, seq_len, hidden_size]
        """
        b, s = name.shape
        # Encode gene names and expressions using ScGPTEmbedding
        name_emb, expr_emb = self.embedding(name, expr)

        # Dual cross-attention layers (parallel execution)
        # Both updates computed simultaneously using current values
        # PyTorch will automatically parallelize independent operations when possible
        for name_layer, expr_layer in zip(self.name_cross_attn_layers, self.expr_cross_attn_layers):
            # Parallel computation: both use current (old) values
            # These operations are independent and can be executed in parallel
            new_name_emb = name_layer(name_emb, expr_emb)
            new_expr_emb = expr_layer(expr_emb, name_emb)
            # Simultaneous update after both computations complete
            name_emb, expr_emb = new_name_emb, new_expr_emb

        # Concatenate CLS token with both sequences: [CLS, name_seq, expr_seq]

        task_cls_token = self.cls_token["task"].expand(b, -1, -1)
        domain_cls_token = self.cls_token["domain"].expand(b, -1, -1)

        aligned_emb = torch.cat([task_cls_token, domain_cls_token, name_emb, expr_emb], dim=1)
        aligned_emb = self.encoder(aligned_emb)  # [b, 2*s + 2, hidden_size]

        split_emb = torch.split(aligned_emb[:, 2:, :], [s, s], dim=1)

        name_emb = split_emb[0] + name_emb
        expr_emb = split_emb[1] + expr_emb

        return aligned_emb, name_emb, expr_emb


class DualTransformerDecoder(nn.Module):
    """
    Minimal Decoder for Dual Transformer.
    Reconstructs gene names (token IDs) and gene expressions (values) sequences.
    """

    def __init__(
        self,
        seq_len: int,
        hidden_size: int,
        dropout: float = 0.1,
        binning: int = None,
    ):
        """
        Args:
            hidden_size: Hidden size for expression reconstruction
            seq_len: Sequence length for name token reconstruction
            dropout: Dropout rate
            binning: Number of bins for discrete expression reconstruction.
                    If None, reconstructs continuous values.
        """
        super().__init__()

        # Build MLP decoder for name reconstruction (token IDs)
        self.decoder_name = nn.Linear(hidden_size, seq_len)

        # Build MLP decoder for expression reconstruction
        if binning is not None:
            # Discrete expression: output bin indices [0, binning]
            self.decoder_expr = nn.Linear(hidden_size, binning + 1)
        else:
            # Continuous expression: output single value
            self.decoder_expr = nn.Sequential(
                nn.Linear(hidden_size, 1),
                Lambda(lambda x: x.squeeze(-1)),
            )

    def forward(self, encoded_name: torch.Tensor, encoded_expr: torch.Tensor):
        """
        Forward pass of MLP decoder to reconstruct name and expr sequences.

        Args:
            encoded_name: Encoded name sequence [batch_size, seq_len, hidden_size]
            encoded_expr: Encoded expression sequence [batch_size, seq_len, hidden_size]

        Returns:
            reconstructed_name: Reconstructed name token logits [b, s, seq_len]
            reconstructed_expr: Reconstructed expression values
                - If discrete: [b, s, binning+1] (logits for bin classification)
                - If continuous: [b, s, seq_len] (expression values, squeezed from [b, s, seq_len, 1])
        """
        reconstructed_name = self.decoder_name(encoded_name)  # [b, s, hidden_size] -> [b, s, seq_len]
        # [b, s, seq_len] or [b, s, binning+1]
        reconstructed_expr = self.decoder_expr(encoded_expr)  # [b, s, hidden_size] -> [b, s, seq_len]

        return reconstructed_name, reconstructed_expr


if __name__ == "__main__":
    pass
