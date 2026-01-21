import torch
import torch.nn as nn
from transformers.activations import ACT2FN


class MLPEncoder(nn.Module):
    """
    MLP Encoder for vector-encoded data (AnnData format).
    Directly processes floating-point vectors without tokenization.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: list[int] = [512, 256, 128],
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        """
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            activation: Activation function name
        """
        super().__init__()

        # Build hidden layers
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            ACT2FN[activation],
            nn.Dropout(dropout),
        )

        hidden_layers = []
        start_dim = hidden_dims[0]
        for end_dim in hidden_dims[1:]:
            hidden_layers.append(
                nn.Sequential(
                    nn.Linear(start_dim, end_dim),
                    ACT2FN[activation],
                    nn.Dropout(dropout),
                )
            )
            start_dim = end_dim
        self.hidden_layers = nn.Sequential(*hidden_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, -1)

        Returns:
            Output tensor of shape (batch_size, hidden_dims[-1])
        """
        x = self.proj(x)
        x = self.hidden_layers(x)
        return x


class MLPDecoder(nn.Module):
    """
    MLP Decoder for reconstructing original input from encoded features.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 1024,
        activation: str = "gelu",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            ACT2FN[activation],
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x
