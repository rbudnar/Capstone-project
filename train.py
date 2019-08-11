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
from main_v2 import generate_dataframe_from_csv_horizontal, generate_dataframe_from_csv_vertical, get_model_inputs, create_multi_generator, build_model
from math import ceil

import wandb
from wandb.keras import WandbCallback

# https://github.com/keras-team/keras/issues/4161#issuecomment-366031228
# https://github.com/tensorflow/tensorflow/issues/7072

# This code appears to resolve the error: failed to create cublas handle: CUBLAS_STATUS_ALLOC_FAILED
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

#project="testing-ml"

VAL_SPLIT = 0.25
EPOCHS = 20
HEIGHT = 224
WIDTH = 224
BATCH_SIZE = 32
# WORKERS = 2

class TrainingRunner:
	def __init__(self, filename="train_1.csv", epochs=EPOCHS, batch_size=BATCH_SIZE, val_split=VAL_SPLIT, use_wandb=False):
		self.use_wandb = use_wandb
		if self.use_wandb:
			print("INITIALIZING WANDB")
			wandb.init()
			
		self.batch_size = batch_size
		self.epochs = epochs
		# df = generate_dataframe_from_csv_vertical(filename, 1)
		df = pd.read_csv(filename,  dtype={'sirna': object})
		# (images, trainY) = get_model_inputs(df)
		train_size = int(len(df)*(1-val_split))
		valid_size = int(len(df)*(val_split))
		self.steps = ceil(train_size/batch_size)
		self.steps_valid = ceil(valid_size/batch_size)

		train_datagen = ImageDataGenerator(
				rescale=1./255,
				validation_split=val_split)


		self.train_generator = train_datagen.flow_from_dataframe(
				df,
				directory="./",
				x_col="img_path",
				y_col="sirna",
				target_size=(HEIGHT, WIDTH),
				batch_size=self.batch_size,
				subset="training",
				class_mode='categorical')

		self.valid_generator = train_datagen.flow_from_dataframe(
				df,
				directory="./",
				x_col="img_path",
				y_col="sirna",
				target_size=(HEIGHT, WIDTH),
				batch_size=self.batch_size,
				subset="validation",
				class_mode='categorical')
	
		# train_generator = create_multi_generator(df, train_datagen, "training") 
		# valid_generator = create_multi_generator(df, train_datagen, "validation") 
		self.setup_callbacks()

		self.model = build_model(height=HEIGHT, width=WIDTH)
		self.model.summary()

	def setup_callbacks(self): 
		logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
		tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
		
		reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, 
								verbose=1, mode='auto', min_delta=0.0001)
		early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=15)

		csv_logger = CSVLogger(filename='./training_log.csv',
                       separator=',',
                       append=True)
		
		checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', verbose=1, save_best_only=True)
		self.callbacks = [checkpointer, reduceLROnPlat, early, csv_logger, tensorboard_callback]

		if self.use_wandb:
			self.callbacks.append(WandbCallback())

	def run(self):		
		model.fit_generator(self.train_generator, 
							steps_per_epoch=self.steps,
							validation_data=self.valid_generator,
							validation_steps=self.steps_valid, 
							# workers=WORKERS, use_multiprocessing=True,
							epochs=self.epochs, callbacks=self.callbacks, verbose=1)

