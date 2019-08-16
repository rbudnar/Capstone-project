from train import TrainingRunner

# runner = TrainingRunner(filename="train_root.csv", use_wandb=True, use_multi=True, use_tb=False)
# runner.run()

runner = TrainingRunner(use_wandb=True, use_tb=False)
runner.run()
