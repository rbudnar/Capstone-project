import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display  # Allows the use of display() for DataFrames
import seaborn as sb
import keras
from keras import optimizers, models, layers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Add, Concatenate, BatchNormalization, GlobalAveragePooling2D

from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
import timeit
import os
import tensorflow as tf

from keras.applications.resnet50 import ResNet50
# from keras.applications.resnet import ResNet152
# from keras.applications.resnet_v2 import ResNet152V2
# from keras.applications.DenseNet121 import DenseNet121
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

HEIGHT = 224
WIDTH = 224
INPUTS = 6


"""
This file contains the main logic for how the neural network is built.
There are a number of experiments and usages here.
"""


def create_multi_generator(df, train_datagen, subset):
    gens = []
    for i in range(1, INPUTS + 1):
        gens.append(train_datagen.flow_from_dataframe(
            df,
            directory="./",
            x_col=f"img_path_{i}",
            y_col="sirna",
            target_size=(HEIGHT, WIDTH),
            batch_size=32,
            subset=subset,
            class_mode='categorical'))

    while True:
        next_set = [gen.next() for gen in gens]
        yield [x[0] for x in next_set], next_set[0][1]


def build_cnn_layer(i, shape=(HEIGHT, WIDTH, 3,)):
    name = f"inputlayer_{i}"
    inputlayer = Input(shape=shape, name=name)

    x = Conv2D(filters=64, kernel_size=5, padding='same',
               kernel_initializer="he_uniform", activation='relu')(inputlayer)
    x = BatchNormalization(name=f"bn_cnn_2_{i}")(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Conv2D(filters=128, kernel_size=3, padding='same',
               kernel_initializer="he_uniform", activation='relu')(x)
    x = BatchNormalization(name=f"bn_cnn_3_{i}")(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same',
               kernel_initializer="he_uniform", activation='relu')(x)
    x = BatchNormalization(name=f"bn_cnn_4_{i}")(x)
    x = MaxPooling2D(pool_size=2)(x)

    x = Conv2D(filters=512, kernel_size=3, padding='same',
               kernel_initializer="he_uniform", activation='relu')(x)
    x = BatchNormalization(name=f"bn_cnn_5_{i}")(x)
    x = GlobalAveragePooling2D()(x)

    model = Model(inputs=inputlayer, outputs=x)
    return model


def build_sequential_layer(previous_layers):
    combined = Concatenate()([x.output for x in previous_layers])
    combined = Dense(1000, kernel_regularizer=l2(0.001), activation="relu")(combined)
    combined = BatchNormalization(name="batch_norm_1")(combined)
    combined = Dropout(0.3)(combined)
    # combined = Activation("relu", name="act_layer")(combined)
    z = Dense(1108, kernel_regularizer=l2(0.001), activation="softmax")(combined)
    return z


def build_sequential_layer_controls(previous_layers, use_mlp=False):
    if use_mlp:
        mlp = create_mlp()
        previous_layers.append(mlp)

    combined = Concatenate()([x.output for x in previous_layers])
    # combined = Dense(240, kernel_regularizer=l2(0.001), activation="relu")(combined)
    # combined = BatchNormalization()(combined)
    # combined = Dropout(0.3)(combined)
    # combined = Dense(120, kernel_regularizer=l2(0.001), activation="relu")(combined)
    # combined = BatchNormalization()(combined)
    # combined = Dropout(0.3)(combined)
    combined = Dense(60, kernel_regularizer=l2(0.001), activation="relu")(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.3)(combined)
    z = Dense(31, activation="softmax")(combined)
    return z


def ConvBlock(n_conv, n_out, shape, x, is_last=False, name=None):
    for i in range(n_conv):
        if name is not None:
            # use of he_uniform as suggested by: https://towardsdatascience.com/why-default-cnn-are-broken-in-keras-and-how-to-fix-them-ce295e5e5f2
            x = Conv2D(n_out, shape, padding='same', name=name, kernel_initializer="he_uniform", activation="relu")(x)
            x = BatchNormalization(name=f"bn_{name}")(x)
        else:
            x = Conv2D(n_out, shape, padding='same', kernel_initializer="he_uniform", activation="relu")(x)
            x = BatchNormalization()(x)

    if is_last:
        out = GlobalAveragePooling2D()(x)
    else:
        out = MaxPooling2D()(x)

    return out


def build_cnn_layer_vgg(i, shape=(HEIGHT, WIDTH, 3,)):
    inputlayer = Input(shape=shape)
    x = ConvBlock(1, 32, (5, 5), inputlayer)
    x = ConvBlock(1, 64, (3, 3), x)
    x = ConvBlock(1, 128, (3, 3), x)
    x = ConvBlock(1, 256, (3, 3), x)
    x = ConvBlock(1, 512, (3, 3), x, is_last=True)

    x = Dense(1000, kernel_regularizer=l2(0.001), activation="relu")(x)
    x = BatchNormalization(name="bn_fc_1")(x)
    # x = Dropout(0.3)(x)
    x = Dense(1108, kernel_regularizer=l2(0.001), activation='softmax')(x)
    model = Model(inputlayer, x)
    return model


def build_vgg_model(shape=(HEIGHT, WIDTH, 3,)):
    # https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.3-using-a-pretrained-convnet.ipynb
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=shape)
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    base_model.trainable = False

    for layer in base_model.layers:
        if layer.name.find("block5") >= 0 or layer.name.find("block4") >= 0:
            layer.trainable = True
        else:
            layer.trainable = False

    model.add(Dense(1000, activation="relu"))
    model.add(BatchNormalization(name="bn_fc_1"))
    model.add(Dropout(0.3))
    # kernel_regularizer=l2(0.001),
    model.add(Dense(1108, activation='softmax'))

    optimizer = optimizers.Adam()
    model.compile(optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    return model


def build_resnet_model(shape=(HEIGHT, WIDTH, 3,)):
    # https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.3-using-a-pretrained-convnet.ipynb
    base_model = ResNet50(weights='imagenet',
                          include_top=False, input_shape=shape)
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    base_model.trainable = False

    model.add(Dense(1000, activation="relu"))
    model.add(BatchNormalization(name="bn_fc_1"))
    model.add(Dropout(0.3))
    # kernel_regularizer=l2(0.001),
    model.add(Dense(1108, activation='softmax'))

    optimizer = optimizers.Adam()
    model.compile(optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    return model


def build_cnn_layer_3(i, shape=(HEIGHT, WIDTH, 3,)):
    inputlayer = Input(shape=shape)
    x = ConvBlock(1, 16, (3, 3), inputlayer)
    x = ConvBlock(1, 32, (3, 3), x)
    x = ConvBlock(1, 64, (3, 3), x)
    x = ConvBlock(1, 128, (3, 3), x)
    x = ConvBlock(1, 256, (3, 3), x)
    x = ConvBlock(1, 512, (3, 3), x, is_last=True)
    # x = ConvBlock(1, 1024, (3, 3), x, is_last=True)
    # x = Flatten()(x)
    model = Model(inputlayer, x)
    return model


def build_model(height=HEIGHT, width=WIDTH):
    model = build_cnn_layer_vgg(1, shape=(height, width, 3))
    optimizer = optimizers.Adam()
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_multi_model(height=HEIGHT, width=WIDTH, controls_only=False, use_mlp=False):
    cnn_layers = []
    for i in range(0, INPUTS):
        layer = build_cnn_layer_3(i, shape=(height, width, 3))
        cnn_layers.append(layer)

    if controls_only:
        output_layer = build_sequential_layer_controls(cnn_layers, use_mlp=use_mlp)
    else:
        output_layer = build_sequential_layer(cnn_layers)

    model = Model(inputs=[x.input for x in cnn_layers], outputs=output_layer)
    optimizer = optimizers.Adam()

    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_cell_classifier_model():
    cnn_layers = []
    for i in range(0, INPUTS):
        layer = build_cnn_layer_3(i)
        cnn_layers.append(layer)

    combined = Concatenate()([x.output for x in cnn_layers])

    combined = Dense(10, kernel_regularizer=l2(0.001), activation="relu")(combined)
    combined = BatchNormalization(name="batch_norm_1")(combined)
    combined = Dropout(0.3)(combined)
    output_layer = Dense(4, kernel_regularizer=l2(0.001), activation="softmax")(combined)

    model = Model(inputs=[x.input for x in cnn_layers], outputs=output_layer)
    optimizer = optimizers.Nadam()
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_cell_multi_model(cell_model_path, height=HEIGHT, width=WIDTH):
    base_model = models.load_model(cell_model_path)
    base_model.trainable = False

    # Pop off concatenate - dense_2 layers, leaving at flattened
    base_model.layers.pop()  # dense_2
    base_model.layers.pop()  # dropout_1
    base_model.layers.pop()  # batch_norm_1
    base_model.layers.pop()  # dense_1
    base_model.layers.pop()  # concatenate_1

    # pop off 6 flatten layers
    base_model.layers.pop()
    base_model.layers.pop()
    base_model.layers.pop()
    base_model.layers.pop()
    base_model.layers.pop()
    base_model.layers.pop()

    layers = [build_connected_layers(base_model.get_layer(
        f"max_pooling2d_{i*3}").output, name=f"cv_block_sirna_{i}") for i in range(1, 7)]

    combined = Concatenate()([x for x in layers])

    combined = Dense(1000, kernel_regularizer=l2(
        0.001), activation="relu")(combined)
    combined = BatchNormalization(name="batch_norm_1")(combined)
    combined = Dropout(0.3)(combined)
    output_layer = Dense(1108, kernel_regularizer=l2(
        0.001), activation="softmax")(combined)

    model = Model(inputs=base_model.input, outputs=output_layer)
    optimizer = optimizers.Nadam()
    model.compile(optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # model.summary()
    return model


def build_transfer_multi_model(model_path, locked_layers=0, height=HEIGHT, width=WIDTH):
    base_model = models.load_model(model_path)
    # base_model.trainable = False

    # Pop off concatenate - dense_2 layers, leaving at flattened
    base_model.layers.pop()  # dense_2
    base_model.layers.pop()  # dropout_1
    base_model.layers.pop()  # batch_norm_1
    base_model.layers.pop()  # dense_1

    for layer in base_model.layers[:locked_layers]:
        layer.trainable = False

    combined = Dense(1000, name="x_dense_1", kernel_regularizer=l2(0.001), activation="relu")(base_model.layers[-1].output)
    combined = BatchNormalization(name="x_batch_norm_1")(combined)
    combined = Dropout(0.3, name="x_dropout_1")(combined)
    output_layer = Dense(1108, name="x_dense_2", kernel_regularizer=l2(0.001), activation="softmax")(combined)

    model = Model(inputs=base_model.input, outputs=output_layer)
    model.compile(optimizers.Nadam(), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def build_connected_layers(prev_output, name):
    x = ConvBlock(1, 32, (5, 5), prev_output, name=f"{name}_1")
    x = ConvBlock(1, 64, (3, 3), x, name=f"{name}_2")
    x = ConvBlock(1, 128, (3, 3), x, name=f"{name}_3")
    x = Flatten()(x)

    return x


def create_mlp():
    # define our MLP network
    model = Sequential()
    model.add(Dense(8, input_dim=(4), kernel_regularizer=l2(
        0.001), activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(4, kernel_regularizer=l2(0.001), activation="relu"))

    # return our model
    return model
