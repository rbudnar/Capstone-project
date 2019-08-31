from train import TrainingRunner
from training_mode import TrainingMode

# runner = TrainingRunner(filename="train_root.csv", use_wandb=True, use_multi=True, use_tb=False)
# runner.run()

# runner = TrainingRunner(use_wandb=True, use_tb=False)
# runner = TrainingRunner(filename="train_controls_root.csv", batch_size=16, controls_only=True, train_mode=TrainingMode.MULTI, use_tb=False, use_wandb=True)
# runner = TrainingRunner(filename="train_controls_root.csv", 
#     batch_size=8, 
#     controls_only=True, 
#     train_mode=TrainingMode.MULTI, 
#     use_tb=False, 
#     # model_path="saved_models/weights.20190822-220829=157=2.57.hdf5",
#     # model_path="saved_models/weights.20190825-104807=12=2.98.hdf5",    
#     use_wandb=True
#     )
# runner.summary()
runner = TrainingRunner(
    train_mode=TrainingMode.SINGLE, 
    use_tb=False, 
    use_wandb=True
    )
runner.run()

# wandb run python train-wb.py