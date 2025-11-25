# train/config.py
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Data
    data_dir: str = "output"
    n_files: int = 1000
    num_rays: int = 100

    # Model
    hidden_dims: list = None
    embedding_dim: int = 64

    # Training
    batch_size: int = 256
    num_epochs: int = 50
    learning_rate: float = 0.001
    margin: float = 1.0
    vis_interval: int = 5
    save_embeddings: bool = False

    # Evaluation
    distance_threshold: float = 5.0
    temporal_window: int = 5
    far_threshold: int = 50

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]