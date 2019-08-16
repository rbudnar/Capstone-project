import numpy as np
import pandas as pd
from keras.utils import Sequence, to_categorical
import cv2
from sklearn.utils import class_weight, shuffle

## https://www.kaggle.com/chandyalex/recursion-cellular-keras-densenet
class MultiGenerator(Sequence):
    def __init__(self, image_filenames, labels,
                 batch_size, is_train=True,
                 mix=False, augment=False, do_one_hot=False):
        self.image_filenames = image_filenames

        if do_one_hot: 
            self.labels = pd.get_dummies(labels)
        else:
            self.labels = to_categorical(labels, 1108)

        self.batch_size = batch_size
        self.is_train = is_train
        self.is_augment = augment
        if(self.is_train):
            self.on_epoch_end()
        self.is_mix = mix

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        if(self.is_train):
            return self.train_generate(batch_x, batch_y)
        return self.valid_generate(batch_x, batch_y)

    def on_epoch_end(self):
        if(self.is_train):
            self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)
        else:
            pass
    
    def mix_up(self, x, y):
        lam = np.random.beta(0.2, 0.4)
        ori_index = np.arange(int(len(x)))
        index_array = np.arange(int(len(x)))
        np.random.shuffle(index_array)        
        
        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]
        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]
        
        return mixed_x, mixed_y

    def train_generate(self, batch_x, batch_y):
        batch_images = []
        # print(batch_x, batch_y)
        for (sample, label) in zip(batch_x, batch_y.transpose()):
            # print(sample, label)
            imgs = []
            for i in range(1, 7):
                img = cv2.resize(cv2.imread(f"{sample}_w{i}.png"), (224,224))
                
                # if(self.is_augment):
                #     img = seq.augment_image(img)
                imgs.append(img)
            
            batch_images.append(imgs)
        batch_images = np.transpose(np.array(batch_images, np.float32)/255, axes=(1,0,2,3,4))
        batch_y = np.array(batch_y, np.float32)
        if(self.is_mix):
            batch_images, batch_y = self.mix_up(batch_images, batch_y)
        # return [x for x in batch_images], to_categorical(batch_y, 1108)
        # print("train", batch_y, batch_y.shape, batch_images.shape)
        return [x for x in batch_images], batch_y

    def valid_generate(self, batch_x, batch_y):
        batch_images = []

        for (sample, label) in zip(batch_x, batch_y.transpose()):
            imgs = []
            for i in range(1, 7):
                img = cv2.imread(f"{sample}_w{i}.png")

                imgs.append(img)            
            batch_images.append(imgs)
        batch_images = np.transpose(np.array(batch_images, np.float32)/255, axes=(1,0,2,3,4))
        
        batch_y = np.array(batch_y, np.float32)
        # print("valid", batch_y, batch_y.shape, batch_images.shape)
        return [x for x in batch_images], batch_y