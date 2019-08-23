from train import TrainingRunner
from training_mode import TrainingMode

# runner = TrainingRunner(filename="train_root.csv", use_wandb=True, use_multi=True, use_tb=False)
# runner.run()

# runner = TrainingRunner(use_wandb=True, use_tb=False)
# runner = TrainingRunner(filename="train_controls_root.csv", batch_size=16, controls_only=True, train_mode=TrainingMode.MULTI, use_tb=False, use_wandb=True)
runner = TrainingRunner(filename="train_controls_root.csv", batch_size=16, controls_only=True, train_mode=TrainingMode.MULTI, use_tb=False, use_wandb=True)
runner.summary()

runner.run()
