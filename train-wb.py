from train import TrainingRunner
from enums import TrainingMode

"""
This file runs the model from the command prompt while logging to wandb.
Use the following command:

wandb run python train-wb.py
"""
runner = TrainingRunner(use_tb=False, use_wandb=True)
runner.summary()

runner.run()
