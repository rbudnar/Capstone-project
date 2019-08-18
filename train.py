import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display # Allows the use of display() for DataFrames
import seaborn as sb

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
		get_model_inputs, create_multi_generator, build_model, build_multi_model, build_vgg_model, build_resnet_model, build_cell_classifier_model, build_cell_multi_model)
from math import ceil
from data_generator_from_kaggle import MultiGenerator
import wandb
from wandb.keras import WandbCallback
from sklearn.model_selection import train_test_split
from training_mode import TrainingMode
from lr_scheduler import WarmUpLearningRateScheduler, WarmUpCosineDecayScheduler
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


CELL_MODEL_PATH = "saved_models\cell_model.h5"


class TrainingRunner:
	def __init__(self, train_mode, filename="train_1.csv", controls_only=False, cell_model_path=CELL_MODEL_PATH, epochs=EPOCHS, batch_size=BATCH_SIZE, val_split=VAL_SPLIT, use_wandb=False, use_tb=True):
		self.use_wandb = use_wandb
		self.use_tb = use_tb
		self.valid_split = val_split
		self.controls_only = controls_only
		self.train_mode = train_mode
		self.time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

		if self.use_wandb:
			print("INITIALIZING WANDB")
			wandb.init()
			
		self.batch_size = batch_size
		self.epochs = epochs

		df = pd.read_csv(filename,  dtype={'sirna': object})

		if train_mode is TrainingMode.MULTI:
			self.create_multi_model(df)
		elif train_mode is TrainingMode.CELL_ONLY:
			self.create_cell_model(df)
		elif train_mode is TrainingMode.CELL_MULTI:
			self.create_multi_model(df, use_cell_model=True, cell_model_path=cell_model_path)
		else:
			self.create_simple_model(df)

	def summary(self):
		if(self.model):
			self.model.summary()

	def create_multi_model(self, df, use_cell_model=False, cell_model_path=CELL_MODEL_PATH):
		''' Creates a model that trains on each of set of 6 images. Can use the cell classifier model for transfer learning.
		'''
		x = df["img_path_root"]
		y = df["sirna"]
		train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.2, stratify=y, random_state=10)

		if self.controls_only: class_count = 31
		else: class_count = 1108

		self.train_generator = MultiGenerator(train_x, train_y, self.batch_size, class_count=class_count, is_train=True)
		self.valid_generator = MultiGenerator(valid_x, valid_y, self.batch_size, class_count=class_count,)
		self.steps = self.train_generator.__len__()
		self.steps_valid = self.valid_generator.__len__()
		if use_cell_model:
			self.model = build_cell_multi_model(cell_model_path, height=HEIGHT, width=WIDTH)
		else:
			self.model = build_multi_model(height=HEIGHT, width=WIDTH, controls_only=self.controls_only)

	def create_simple_model(self, df):
		''' Creates a model that trains on a single cell stain types (1 of 6 stain images). Does not use cell classifier.
		'''
		train_datagen = ImageDataGenerator(
			rescale=1./255,
			horizontal_flip=True,
			vertical_flip=True,
			validation_split=self.valid_split)	
	
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
		# self.model = build_resnet_model()

	def create_cell_model(self, df):
    	# ''' Creates a model that only learns our 4 cell types. Does not classify treatments (siRNAs)
		# '''
		x = df["img_path_root"]
		y = df["cell_type"]

		train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.2, stratify=y, random_state=10)
		self.train_generator = MultiGenerator(train_x, train_y, batch_size, is_train=True, do_one_hot=True)
		self.valid_generator = MultiGenerator(valid_x, valid_y, batch_size, do_one_hot=True)
		self.steps = self.train_generator.__len__()
		self.steps_valid = self.valid_generator.__len__()
		self.model = build_cell_classifier_model()

	def run(self):
		logdir = os.path.join("logs", self.time_stamp)

		callbacks = []

		# callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='auto', min_delta=0.0001))

		total_steps = int(self.epochs * self.train_generator.__len__())
		warmup_epoch = 5
		warmup_steps = int(warmup_epoch * self.train_generator.__len__())
		warmup_batches = warmup_epoch * self.train_generator.__len__()

		callbacks.append(WarmUpCosineDecayScheduler(learning_rate_base=0.0005,
                                        total_steps=total_steps,
                                        warmup_learning_rate=0.0,
                                        warmup_steps=warmup_steps,
                                        hold_base_rate_steps=0, verbose=0))
		callbacks.append(EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=15))

		callbacks.append(CSVLogger(filename='./training_log.csv',
                       separator=',',
                       append=True))
		
		if self.train_mode is TrainingMode.CELL_ONLY:
			checkpoint_path = f"saved_models/cell_model.weights.{self.time_stamp}.hdf5"
		else:
			checkpoint_path = f"saved_models/weights.{self.time_stamp}.hdf5"

		print("checkpoint path: ", checkpoint_path)
		callbacks.append(ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True))
		
		# callbacks = [checkpointer, reduceLROnPlat, early, csv_logger, tensorboard_callback]
		if self.use_wandb:
			callbacks.append(WandbCallback())
		if self.use_tb:
			callbacks.append(keras.callbacks.TensorBoard(log_dir=logdir))	#, write_grads=True, histogram_freq=1		

		## workaround for `FailedPreconditionError: Attempting to use uninitialized value Adam/lr`
		keras.backend.get_session().run(tf.global_variables_initializer())

		self.model.fit_generator(self.train_generator, 
							steps_per_epoch= self.steps,
							validation_data=self.valid_generator,
							validation_steps= self.steps_valid, 
							# workers=WORKERS, use_multiprocessing=True,
							epochs=self.epochs, callbacks=callbacks, verbose=1)

