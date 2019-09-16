import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display  # Allows the use of display() for DataFrames
import seaborn as sb

import keras
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Add, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

import timeit
import os
from datetime import datetime
import tensorflow as tf
from main_v2 import (create_multi_generator, build_model, build_multi_model, build_vgg_model,
                     build_resnet_model, build_cell_classifier_model, build_cell_multi_model, build_transfer_multi_model)
from preprocessing import (generate_dataframe_from_csv_horizontal, generate_dataframe_from_csv_vertical,
                           get_model_inputs)
from math import ceil
from data_generator_from_kaggle import MultiGenerator, TestMultiGenerator
import wandb
from wandb.keras import WandbCallback
from sklearn.model_selection import train_test_split
from enums import TrainingMode, CellType
from lr_scheduler import WarmUpLearningRateScheduler, WarmUpCosineDecayScheduler
from lr_logger import LRTensorBoard
# https://github.com/keras-team/keras/issues/4161#issuecomment-366031228
# https://github.com/tensorflow/tensorflow/issues/7072

# This code appears to resolve the error: failed to create cublas handle: CUBLAS_STATUS_ALLOC_FAILED
# from keras.backend.tensorflow_backend import set_session

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# #config.log_device_placement = True  # to log device placement (on which device the operation ran)
#                                     # (nothing gets printed in Jupyter, only if you run it standalone)
# sess = tf.Session(config=config)
# set_session(sess)  # set this TensorFlow session as the default session for Keras

VAL_SPLIT = 0.2
EPOCHS = 300
HEIGHT = 224
WIDTH = 224
BATCH_SIZE = 32

CELL_MODEL_PATH = "saved_models\cell_model.h5"


class TrainingRunner:
    def __init__(self, train_mode=TrainingMode.MULTI,
                 filename="train_1.csv",
                 controls_only=False,
                 cell_model_path=CELL_MODEL_PATH,
                 epochs=EPOCHS,
                 batch_size=BATCH_SIZE,
                 val_split=VAL_SPLIT,
                 use_wandb=False,
                 use_tb=True,
                 model_path=None,
                 use_mlp=False,
                 height=HEIGHT,
                 width=WIDTH,
                 resume=False,
                 locked_layers=0,
                 cell_type="ALL"):

        self.use_wandb = use_wandb
        self.use_tb = use_tb
        self.resume = resume

        if self.use_wandb:
            print("INITIALIZING WANDB")
            if resume:
                wandb.init(resume=True)
            else:
                wandb.init()
            self.filename = wandb.config.filename
            self.valid_split = wandb.config.val_split
            self.controls_only = wandb.config.controls_only
            self.train_mode = wandb.config.train_mode
            self.batch_size = wandb.config.batch_size
            self.epochs = wandb.config.epochs
            self.height = wandb.config.height
            self.width = wandb.config.width
            self.cell_type = wandb.config.cell_type
            self.model_path = wandb.config.model_path
            self.locked_layers = wandb.config.locked_layers
        else:
            self.filename = filename
            self.valid_split = val_split
            self.controls_only = controls_only
            self.train_mode = train_mode
            self.batch_size = batch_size
            self.epochs = epochs
            self.height = HEIGHT
            self.width = WIDTH
            self.cell_type = cell_type
            self.model_path = model_path
            self.locked_layers = locked_layers

        print(
            self.valid_split,
            self.controls_only,
            self.train_mode,
            self.batch_size,
            self.epochs,
            self.cell_type,
            self.height,
            self.width,
            self.model_path
        )

        self.time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.initial_epoch = 0

        df = pd.read_csv(self.filename,  dtype={'sirna': object})

        if train_mode is TrainingMode.MULTI:
            self.create_multi_model(df, use_mlp=use_mlp)
        elif train_mode is TrainingMode.CELL_ONLY:
            self.create_cell_model(df)
        elif train_mode is TrainingMode.CELL_MULTI:
            self.create_multi_model(df, use_mlp=use_mlp,
                                    use_cell_model=True, cell_model_path=cell_model_path)
        elif train_mode is TrainingMode.TEST_ONLY:
            assert(self.model_path is not None)
            self.model = load_model(self.model_path)
        else:
            self.create_simple_model(df)

    def summary(self):
        if(self.model):
            self.model.summary()

    def create_multi_model(self, df, use_cell_model=False, cell_model_path=CELL_MODEL_PATH, use_mlp=False):
        ''' Creates a model that trains on each of set of 6 images. Can use the cell classifier model for transfer learning.
        '''
        if self.cell_type is not None and self.cell_type is not "ALL":
            rows = df[df["cell_type"] == self.cell_type]
            x = rows[["img_path_root", "cell_type"]]
            y = rows["sirna"]
        else:
            x = df[["img_path_root", "cell_type"]]
            y = df["sirna"]

        self.train_x, self.valid_x, self.train_y, self.valid_y = train_test_split(
            x, y, test_size=0.2, stratify=y, random_state=10)

        if self.controls_only:
            print("CONTROLS ONLY: 31")
            class_count = 31
        else:
            print("FULL DATASET: 1108")
            class_count = 1108

        self.train_generator = MultiGenerator(
            self.train_x, self.train_y, self.batch_size, class_count=class_count, augment=True, is_train=True, height=self.height, width=self.width)
        self.valid_generator = MultiGenerator(
            self.valid_x, self.valid_y, self.batch_size, class_count=class_count, is_train=False, height=self.height, width=self.width)
        self.steps = self.train_generator.__len__()
        self.steps_valid = self.valid_generator.__len__()
        if use_cell_model:
            print("creating new cell model")
            self.model = build_cell_multi_model(
                cell_model_path, height=self.height, width=self.width)
        else:
            if self.resume:
                print(f"loading model from {self.model_path}")
                self.model = keras.models.load_model(self.model_path)
                if use_wandb:
                    self.initial_epoch = wandb.run.step
                else:
                    self.initial_epoch = int(
                        self.model_path[self.model_path.find("=")+1:self.model_path.rfind("=")])
                print(self.initial_epoch)
            else:
                if self.model_path is not None:
                    print("creating transfer multi model")
                    self.model = build_transfer_multi_model(
                        self.model_path, locked_layers=self.locked_layers, height=self.height, width=self.width)
                else:
                    print("creating new multi model")
                    self.model = build_multi_model(
                        height=self.height, width=self.width, controls_only=self.controls_only)

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
            x_col="img_rgb",
            y_col="sirna",
            target_size=(self.height, self.width),
            batch_size=self.batch_size,
            subset="training",
            class_mode='categorical')

        self.valid_generator = train_datagen.flow_from_dataframe(
            df,
            directory="./",
            x_col="img_rgb",
            y_col="sirna",
            target_size=(self.height, self.width),
            batch_size=self.batch_size,
            subset="validation",
            class_mode='categorical')

        self.steps = ceil(self.train_generator.n /
                          self.train_generator.batch_size)
        self.steps_valid = ceil(
            self.valid_generator.n/self.valid_generator.batch_size)
        self.model = build_model(height=self.height, width=self.width)

    def create_cell_model(self, df):
        # ''' Creates a model that only learns our 4 cell types. Does not classify treatments (siRNAs)
        # '''
        x = df["img_path_root"]
        y = df["cell_type"]

        train_x, valid_x, train_y, valid_y = train_test_split(
            x, y, test_size=0.2, stratify=y, random_state=10)
        self.train_generator = MultiGenerator(
            train_x, train_y, batch_size, is_train=True, do_one_hot=True)
        self.valid_generator = MultiGenerator(
            valid_x, valid_y, batch_size, do_one_hot=True)
        self.steps = self.train_generator.__len__()
        self.steps_valid = self.valid_generator.__len__()
        self.model = build_cell_classifier_model()

    def predict(self, filepath):
        assert(self.model is not None)
        df = pd.read_csv(filepath)
        if self.cell_type is not None:
            df = df[df["cell_type"] == self.cell_type].reset_index(drop=True)

        x = df["img_path_root"]

        self.test_generator = TestMultiGenerator(x, self.batch_size)
        self.steps = self.test_generator.__len__()
        # generate predictions
        results = self.model.predict_generator(
            self.test_generator, steps=self.steps, verbose=1)

        # process predictions
        predictions = self.prepare_results_from_predictions(results, df)

        self.save_prediction_file(
            predictions, f"predictions.{self.cell_type}.csv")

        return predictions

    def prepare_results_from_predictions(self, results, original_df):
        df = pd.DataFrame(results)
        # find most likely treatment from a given sample and add create a dataframe from it and its p value
        bests = df.apply(lambda x: pd.Series(
            [str(np.argmax(x)), x[np.argmax(x)]]), axis=1)
        # label our columns
        bests.columns = ["sirna", "p"]
        # merge predictions back to original dataframe containing id_code and cell_type
        output = pd.concat([original_df, bests], axis=1)

        # at this point we have two samples for each well; we need to determine the highest probability treatment of these two for submission
        # determine indexes of highest probability treatment
        idxs = output.groupby(["id_code"])["p"].transform(max) == output["p"]
        # filter and return by our best guess treatment
        return output[idxs].reset_index(drop=True)

    def save_prediction_file(self, df, dest_filename):
        data_for_upload = df.filter(items=["id_code", "sirna"]).to_csv(
            dest_filename, index=False)

    def evaluate(self):
        ''' Just for checking model loading for now
        '''
        print(self.model.evaluate_generator(
            self.valid_generator), self.model.metrics_names)

    def run(self):
        logdir = os.path.join("logs", self.time_stamp)

        callbacks = []

        # removed in favor of WarmUpCosineDecayScheduler
        # callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='auto', min_delta=0.0001))

        total_steps = int(self.epochs * self.train_generator.__len__())
        warmup_epoch = 10
        warmup_steps = int(warmup_epoch * self.train_generator.__len__())
        warmup_batches = warmup_epoch * self.train_generator.__len__()
        learning_rate = .1 * self.batch_size / 256
        callbacks.append(WarmUpCosineDecayScheduler(learning_rate_base=learning_rate,
                                                    total_steps=total_steps,
                                                    global_step_init=self.initial_epoch * self.steps,
                                                    warmup_learning_rate=0.0,
                                                    warmup_steps=warmup_steps,
                                                    hold_base_rate_steps=0, verbose=0))

        # removed in favor of WarmUpCosineDecayScheduler
        # # callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=20))

        # removed in favor of wandb
        # callbacks.append(LRTensorBoard(logdir))

        if self.train_mode is TrainingMode.CELL_ONLY:
            modifier = "cell_model."
        elif self.controls_only:
            modifier = "controls."
        else:
            modifier = ""

        checkpoint_path = f"saved_models/weights.{modifier}{self.cell_type}.{self.time_stamp}"

        print("checkpoint path: ", checkpoint_path)
        callbacks.append(ModelCheckpoint(filepath=checkpoint_path +
                                         "={epoch:02d}={val_loss:.2f}.hdf5", verbose=1, monitor='val_loss', save_best_only=True))

        if self.use_wandb:
            callbacks.append(WandbCallback())
        if self.use_tb:
            callbacks.append(keras.callbacks.TensorBoard(log_dir=logdir))

        # ## workaround for `FailedPreconditionError: Attempting to use uninitialized value Adam/lr`
        # # keras.backend.get_session().run(tf.global_variables_initializer())

        self.model.fit_generator(self.train_generator,
                                 steps_per_epoch=self.steps,
                                 validation_data=self.valid_generator,
                                 validation_steps=self.steps_valid,
                                 initial_epoch=self.initial_epoch,
                                 epochs=self.epochs,
                                 callbacks=callbacks,
                                 verbose=1)
