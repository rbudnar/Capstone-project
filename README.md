## Machine Learning Capstone Project

For Kaggle competition: https://www.kaggle.com/c/recursion-cellular-image-classification
Background information: https://www.rxrx.ai/#the-data

The full Dataset for this project is available here: https://www.kaggle.com/c/recursion-cellular-image-classification/data
Download and extract the full dataset to this directory - you should have a test and train folder with subfolders and images.

### Main libraries used for this project:

Keras
sklearn
numpy
pandas
Tensorflow-gpu
cv2
imgaug
wandb (for logging and analysis)

### Running the project

Make sure the above libraries are available. Using wandb is highly encouraged since training variables can
easily be loaded from the `config-defaults.yml` file. However, it can be run by configuring the `TrainingRunner`
in train-wb.py with the desired options.

To run the project using wandb, run this command from the command line after logging in with your wandb account (its free - sign up at www.wandb.com):
`wandb run python train-wb.py`

The CSV files has been preprocessed and re-saved for ease of training and testing. When using the multi-generator, use:
`train_root.csv` for training full data
`test_root.csv` for testing
`train_controls_root.csv` for using training control data
`all_controls_root.csv` for using all control data
