"""
Configuration file for hyperparameters and settings.

Modify these values to experiment with different configurations.
Uses Pydantic for type validation and better structure.
"""

from typing import List
from pydantic import BaseModel, Field


class Config(BaseModel):
    """
    Configuration for the voice-based heart failure prediction challenge.

    This class uses Pydantic for automatic validation and type checking.
    Modify the default values below to experiment with different settings.
    """

    # Data settings
    data_path: str = Field(
        default="data/dataset.parquet",
        description="Path to the dataset parquet file",
    )

    # Model architecture
    embedding_dim: int = Field(
        default=64,
        gt=0,
        description="Dimension of learned embeddings",
    )
    hidden_dims: List[int] = Field(
        default=[256, 128],
        description="Hidden layer dimensions (can add more layers)",
    )
    dropout: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Dropout probability for regularization",
    )

    # Training settings
    batch_size: int = Field(
        default=128,
        gt=0,
        description="Batch size for training",
    )
    num_epochs: int = Field(
        default=50,
        gt=0,
        description="Number of training epochs per LOPO fold",
    )
    learning_rate: float = Field(
        default=0.001,
        gt=0.0,
        description="Learning rate for optimizer",
    )
    weight_decay: float = Field(
        default=0.0,
        ge=0.0,
        description="L2 regularization (0 = no regularization)",
    )

    # Device settings
    use_cuda: bool = Field(
        default=True,
        description="Set to False to force CPU usage",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )

    # Evaluation settings
    save_embeddings: bool = Field(
        default=True,
        description="Save embeddings after training",
    )
    plot_embeddings: bool = Field(
        default=True,
        description="Generate PCA visualization",
    )
    save_model: bool = Field(
        default=False,
        description="Save model checkpoints (can be large)",
    )

    # Advanced settings (for experimentation)
    use_class_weights: bool = Field(
        default=False,
        description="Balance classes in loss function",
    )
    early_stopping: bool = Field(
        default=False,
        description="Stop training if no improvement",
    )
    patience: int = Field(
        default=10,
        gt=0,
        description="Epochs to wait for improvement (if early stopping enabled)",
    )

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        extra = "forbid"  # Prevent adding unexpected fields

    def print_config(self) -> None:
        """Print current configuration in a formatted way."""
        print("\n" + "=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        for field_name, field_info in self.__fields__.items():
            value = getattr(self, field_name)
            print(f"  {field_name}: {value}")
        print("=" * 60 + "\n")


def get_config() -> Config:
    """
    Get the configuration instance.

    Returns
    -------
    Config
        Configuration object with all settings.
    """
    return Config()


if __name__ == "__main__":
    config = get_config()
    config.print_config()
