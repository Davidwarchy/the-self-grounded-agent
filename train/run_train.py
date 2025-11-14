# bitmap/train/run_training.py
from train import TrainingConfig, train

config = TrainingConfig(
    data_dir="output/2025-11-14-213949_random_walk_10k",
    n_files=100,
    num_epochs=5, 
    vis_interval=1
)
train(config)

# python -m train.run_train