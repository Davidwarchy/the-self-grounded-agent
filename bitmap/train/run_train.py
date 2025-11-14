# bitmap/train/run_training.py
from train import TrainingConfig, train

config = TrainingConfig(
    data_dir="output/2025-11-14-111925_random_walk_100k",
    n_files=1000,
    num_epochs=3
)
train(config)