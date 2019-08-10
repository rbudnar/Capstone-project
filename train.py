import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display # Allows the use of display() for DataFrames
import seaborn as sb
# %matplotlib inline

from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Add, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint

import timeit
import os
from datetime import datetime
import tensorflow as tf
from main_v2 import generate_dataframe_from_csv3, get_model_inputs, create_multi_generator, build_model
from math import ceil

import wandb
from wandb.keras import WandbCallback

wandb.init(project="testing-ml")

logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

df = generate_dataframe_from_csv3("train.csv")
(images, trainY) = get_model_inputs(df)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.25)

train_generator = create_multi_generator(df, train_datagen, "training") 
valid_generator = create_multi_generator(df, train_datagen, "validation") 

model = build_model()
model.summary()

epochs = 20
batch_size = 32
steps = ceil(54773/batch_size)
steps_valid = ceil(18257/batch_size)

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', verbose=1, save_best_only=True)
model.fit_generator(train_generator, 
                    steps_per_epoch=steps,
                    validation_data=valid_generator,
                    validation_steps=steps_valid,                    
                    epochs=epochs, callbacks=[checkpointer, tensorboard_callback, WandbCallback()], verbose=1)