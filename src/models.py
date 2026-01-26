"""
Model definitions for learning voice embeddings.

This module contains a simple baseline model. You should modify and extend
this to create better embeddings that capture temporal patterns and
patient-specific voice characteristics.
"""

import torch
import torch.nn as nn


class EmbeddingModel(nn.Module):
    """
    Simple baseline model for learning voice embeddings.
    This is a basic feedforward network that you should improve upon.

    Parameters
    ----------
    input_dim : int
        Dimension of input acoustic features.
    embedding_dim : int, default=64
        Dimension of the learned embedding space.
    hidden_dims : list of int, default=[256, 128]
        Dimensions of hidden layers.
    dropout : float, default=0.3
        Dropout probability for regularization.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 64,
        hidden_dims: list = [256, 128],
        dropout: float = 0.3,
    ):
        super(EmbeddingModel, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Build the network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        # Final embedding layer (no activation - embeddings can be negative)
        layers.append(nn.Linear(prev_dim, embedding_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        embeddings : torch.Tensor
            Embedding tensor of shape (batch_size, embedding_dim).
        """
        return self.encoder(x)

    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LinearClassifierHead(nn.Module):
    """
    Simple linear classifier head.

    This is intentionally kept simple to force the embedding model to learn
    rich, discriminative representations. A complex classifier would defeat
    the purpose of learning good embeddings.

    Parameters
    ----------
    embedding_dim : int
        Dimension of input embeddings.
    num_classes : int, default=2
        Number of output classes.
    """

    def __init__(self, embedding_dim: int, num_classes: int = 2):
        super(LinearClassifierHead, self).__init__()
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.

        Parameters
        ----------
        embeddings : torch.Tensor
            Input embeddings of shape (batch_size, embedding_dim).

        Returns
        -------
        logits : torch.Tensor
            Class logits of shape (batch_size, num_classes).
        """
        return self.linear(embeddings)
