import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display # Allows the use of display() for DataFrames
import seaborn as sb
# %matplotlib inline

import keras
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Add, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

import timeit
import os
from datetime import datetime
import tensorflow as tf
from main_v2 import (generate_dataframe_from_csv_horizontal, generate_dataframe_from_csv_vertical, 
		get_model_inputs, create_multi_generator, build_model, build_multi_model)
from math import ceil
from data_generator_from_kaggle import MultiGenerator
import wandb
from wandb.keras import WandbCallback
from sklearn.model_selection import train_test_split

# https://github.com/keras-team/keras/issues/4161#issuecomment-366031228
# https://github.com/tensorflow/tensorflow/issues/7072

# This code appears to resolve the error: failed to create cublas handle: CUBLAS_STATUS_ALLOC_FAILED
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

VAL_SPLIT = 0.2
EPOCHS = 20
HEIGHT = 224
WIDTH = 224
BATCH_SIZE = 32
# WORKERS = 2

class TrainingRunner:
	def __init__(self, filename="train_1.csv", epochs=EPOCHS, batch_size=BATCH_SIZE, val_split=VAL_SPLIT, use_multi=False, use_wandb=False, use_tb=True):
		self.use_wandb = use_wandb
		self.use_tb = use_tb
		if self.use_wandb:
			print("INITIALIZING WANDB")
			wandb.init()
			
		self.batch_size = batch_size
		self.epochs = epochs
		# df = generate_dataframe_from_csv_vertical(filename, 1)
		df = pd.read_csv(filename,  dtype={'sirna': object})
		# (images, trainY) = get_model_inputs(df)
		# train_size = int(len(df)*(1-val_split))
		# valid_size = int(len(df)*(val_split))
		
		train_datagen = ImageDataGenerator(
				rescale=1./255,
				horizontal_flip=True,
				vertical_flip=True,
				validation_split=val_split)

		if use_multi:
			x = df["img_path_root"]
			y = df["sirna"]
			train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.2, stratify=y, random_state=10)
			self.train_generator = MultiGenerator(train_x, train_y, 32, is_train=True)
			self.valid_generator = MultiGenerator(valid_x, valid_y, 32)
			self.steps = self.train_generator.__len__()
			self.steps_valid = self.valid_generator.__len__()
			self.model = build_multi_model(height=HEIGHT, width=WIDTH)
		else:
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
			self.steps = ceil(self.train_generator.n/self.train_generator.batch_size)
			self.steps_valid = ceil(self.valid_generator.n/self.valid_generator.batch_size)
			self.model = build_model(height=HEIGHT, width=WIDTH)

		self.model.summary()

	# def setup_callbacks(self): 
	# 	logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
	# 	self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
		
	# 	self.reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, 
	# 							verbose=1, mode='auto', min_delta=0.0001)
	# 	self.early = EarlyStopping(monitor="val_loss", 
    #                   mode="min", 
    #                   patience=15)

	# 	self.csv_logger = CSVLogger(filename='./training_log.csv',
    #                    separator=',',
    #                    append=True)
		
	# 	self.checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', verbose=1, save_best_only=True)
	# 	self.callbacks = [self.checkpointer, self.reduceLROnPlat, self.early, self.csv_logger, self.tensorboard_callback]

	# 	if self.use_wandb:
	# 		self.callbacks.append(WandbCallback())

	def run(self):
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
		
		callbacks = [checkpointer, reduceLROnPlat, early, csv_logger, tensorboard_callback]
		if self.use_wandb:
			callbacks.append(WandbCallback())
		if self.use_tb:
			callbacks.append(tensorboard_callback)			

		## workaround for `FailedPreconditionError: Attempting to use uninitialized value Adam/lr`
		keras.backend.get_session().run(tf.global_variables_initializer())

		self.model.fit_generator(self.train_generator, 
							steps_per_epoch= self.steps,
							validation_data=self.valid_generator,
							validation_steps= self.steps_valid, 
							# workers=WORKERS, use_multiprocessing=True,
							epochs=self.epochs, callbacks=callbacks, verbose=1)

