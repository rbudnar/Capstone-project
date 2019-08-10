import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display # Allows the use of display() for DataFrames
import seaborn as sb

# import keras
from tensorflow import keras
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Add, Concatenate, BatchNormalization
# from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
# from keras import optimizers
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential, Model
# from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Add, Concatenate
# from keras.layers.normalization import BatchNormalization
# from keras.callbacks import ModelCheckpoint
import timeit
import os
import tensorflow as tf

logdir = os.path.join("logs")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
HEIGHT = 150
WIDTH = 150
INPUTS = 1
## loading from dataframe https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c

# def generate_dataframe_from_csv(path):
#     data = pd.read_csv(path)
#     columns = (data.apply(lambda r: pd.Series(gen_image_paths(r)), axis=1)
#         .stack()
#         .rename("img_path")
#         .reset_index(level=1, drop=True))
#     data["sirna"] = data["sirna"].apply(lambda s: str(s))
#     return data.join(columns).reset_index(drop=True)

# def gen_image_paths(row):
#     path_root = f"train/{row['experiment']}/Plate{row['plate']}/{row['well']}"
#     return [f"{path_root}_s{site}_w{image}.png" for site in range(1, 3) for image in range(1,7)]


# def generate_dataframe_from_csv2(path):
#     data = pd.read_csv(path)
#     columns = (data.apply(lambda r: pd.Series(gen_image_paths2(r)), axis=1)
#         .stack()
#         .rename("img_path")
#         .reset_index(level=1, drop=True))
#     data["sirna"] = data["sirna"].apply(lambda s: str(s))
#     return data.join(columns).reset_index(drop=True)

def gen_image_paths2(row):
    path_root = f"train/{row['experiment']}/Plate{row['plate']}/{row['well']}"
    return [f"{path_root}_s{site}" for site in range(1, 3)] 

def generate_dataframe_from_csv3(path):
    data = pd.read_csv(path)
    columns = (data.apply(lambda r: pd.Series(gen_image_paths2(r)), axis=1)
        .stack()
        .rename("img_path")
        .reset_index(level=1, drop=True))
    data["sirna"] = data["sirna"].apply(lambda s: str(s))
    data = data.join(columns).reset_index(drop=True)
    
    for i in range(1,1+INPUTS):
        data[f"img_path_{i}"] = data.apply(lambda row: f"{row['img_path']}_w{i}.png", axis=1)
    return data

# def gen_image_paths3(row):
#     path_root = f"train/{row['experiment']}/Plate{row['plate']}/{row['well']}"
#     return [f"{path_root}_s{site}" for site in range(1, 3)] 

def get_model_inputs(df):
    trainY = df["sirna"]
    im_paths = df["img_path"].apply(lambda r: [f"{r}_w{image}.png" for image in range(1,7)])
    splits = np.hsplit(np.stack(np.array(im_paths)), 6)
    
    images = [np.hstack(s) for s in splits]
    
    return (images, trainY) 

def create_multi_generator(df, train_datagen, subset):
    gens = []
    for i in range(1,INPUTS + 1):
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


def build_cnn_layer(i, shape=(HEIGHT,WIDTH,3,)):
    name = f"inputlayer_{i}"
    inputlayer = Input(shape=shape, name=name)
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(inputlayer)
    x = BatchNormalization(name=f"bn_cnn_1_{i}")(x)
    x = MaxPooling2D(pool_size=2)(x)
    
    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization(name=f"bn_cnn_2_{i}")(x)
    x = MaxPooling2D(pool_size=2)(x)
    
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization(name=f"bn_cnn_3_{i}")(x)
    x = MaxPooling2D(pool_size=2)(x)

    # x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    # x = BatchNormalization(name=f"bn_cnn_4_{i}")(x)
    # x = MaxPooling2D(pool_size=2)(x)

    # x = Conv2D(filters=1024, kernel_size=3, padding='same', activation='relu')(x)
    # x = BatchNormalization(name=f"bn_cnn_5_{i}")(x)
    # x = MaxPooling2D(pool_size=2)(x)
    
    # x = Conv2D(filters=64, kernel_size=3, padding='same')(inputlayer)
    # x = Activation("relu")(x)
    # x = BatchNormalization(name=f"bn_cnn_1_{i}")(x)
    # x = MaxPooling2D(pool_size=2)(x)
    
    # x = Conv2D(filters=128, kernel_size=3, padding='same')(x)
    # x = Activation("relu")(x)
    # x = BatchNormalization(name=f"bn_cnn_2_{i}")(x)
    # x = MaxPooling2D(pool_size=2)(x)
    
    # x = Conv2D(filters=256, kernel_size=3, padding='same')(x)
    # x = Activation("relu")(x)
    # x = BatchNormalization(name=f"bn_cnn_3_{i}")(x)
    # x = MaxPooling2D(pool_size=2)(x)
    
    # x = Conv2D(filters=512, kernel_size=3, padding='same')(x)
    # x = Activation("relu")(x)
    # x = BatchNormalization(name=f"bn_cnn_4_{i}")(x)
    # x = MaxPooling2D(pool_size=2)(x)
    
    # x = Conv2D(filters=1024, kernel_size=3, padding='same')(x)
    # x = Activation("relu")(x)
    # x = BatchNormalization(name=f"bn_cnn_5_{i}")(x)
    # x = MaxPooling2D(pool_size=2)(x)

    x = Flatten(name=f"flattener_{i}")(x)
    x = Dense(1108, activation="softmax")(x)
    model = Model(inputs=inputlayer, outputs=x)
    return model

def build_sequential_layer(previous_layers):
    combined = Concatenate()([x.output for x in previous_layers])
    combined = BatchNormalization(name="batch_norm_1")(combined)
    combined = Activation("relu", name="act_layer")(combined)
    z = Dense(2000, activation="softmax")(combined)
    z = Dense(1108, activation="softmax")(combined)
    return z

# def build_model():
#     cnn_layers = []
#     for i in range(0,INPUTS):
#         layer = build_cnn_layer(i)
#         cnn_layers.append(layer)

#     output_layer = build_sequential_layer(cnn_layers)
#     model = Model(inputs=[x.input for x in cnn_layers], outputs=output_layer)
#     optimizer = optimizers.Adam()    
#     model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

def build_model():
    # cnn_layers = []
    # for i in range(0, INPUTS):
    #     layer = build_cnn_layer(i)
    #     cnn_layers.append(layer)
    model = build_cnn_layer(1)
    # output_layer = build_sequential_layer(cnn_layers)
    # combined = BatchNormalization(name="batch_norm_1")(layer)
    # combined = Activation("relu", name="act_layer")(combined)
    # z = Dense(2000, activation="softmax")(combined)
    # z = Dense(1108, activation="softmax")(combined)
    # model = Model(inputs=[x.input for x in cnn_layers], outputs=output_layer)
    optimizer = optimizers.Adam()    
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model