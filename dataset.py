import math
import os
import random

import keras
import numpy as np
from keras.utils import np_utils
from PIL import Image

from utils import cvtColor, preprocess_input, resize_image

class WebFacesDataset(keras.utils.Sequence):
    def __init__(self, input_shape, lines, batch_size, num_classes, random):
        self.input_shape = input_shape
        self.lines = lines
        self.length = len(lines)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.random = random

        self.paths  = []
        self.labels = []

        self.load_dataset()
        
    def __len__(self):
        return math.ceil(self.length / float(self.batch_size))

    def __getitem__(self, index):
        images = np.zeros((self.batch_size // 3, 3, self.input_shape[0], self.input_shape[1], 3))
        labels = np.zeros((self.batch_size // 3, 3))
        
        for i in range(self.batch_size // 3):
            c               = random.randint(0, self.num_classes - 1)
            selected_path   = self.paths[self.labels[:] == c]
            while len(selected_path) < 2:
                c               = random.randint(0, self.num_classes - 1)
                selected_path   = self.paths[self.labels[:] == c]

            image_indexes = np.random.choice(range(0, len(selected_path)), 2)
            image = cvtColor(Image.open(selected_path[image_indexes[0]]))

            if self.rand()<.5 and self.random: 
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image = True)
            image = preprocess_input(np.array(image, dtype='float32'))

            images[i, 0, :, :, :] = image
            labels[i, 0] = c
            
            image = cvtColor(Image.open(selected_path[image_indexes[1]]))

            if self.rand()<.5 and self.random: 
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image = True)
            image = preprocess_input(np.array(image, dtype='float32'))
            
            images[i, 1, :, :, :] = image
            labels[i, 1] = c

            different_c         = list(range(self.num_classes))
            different_c.pop(c)
            
            different_c_index   = np.random.choice(range(0, self.num_classes - 1), 1)
            current_c           = different_c[different_c_index[0]]
            selected_path       = self.paths[self.labels == current_c]
            while len(selected_path) < 1:
                different_c_index   = np.random.choice(range(0, self.num_classes - 1), 1)
                current_c           = different_c[different_c_index[0]]
                selected_path       = self.paths[self.labels == current_c]

            image_indexes       = np.random.choice(range(0, len(selected_path)), 1)
            image               = cvtColor(Image.open(selected_path[image_indexes[0]]))

            if self.rand()<.5 and self.random: 
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image = True)
            image = preprocess_input(np.array(image, dtype='float32'))
            
            images[i, 2, :, :, :] = image
            labels[i, 2] = current_c

        images1 = np.array(images)[:, 0, :, :, :]
        images2 = np.array(images)[:, 1, :, :, :]
        images3 = np.array(images)[:, 2, :, :, :]
        images = np.concatenate([images1, images2, images3], 0)
        
        labels1 = np.array(labels)[:, 0]
        labels2 = np.array(labels)[:, 1]
        labels3 = np.array(labels)[:, 2]
        labels = np.concatenate([labels1, labels2, labels3], 0)

        labels = np_utils.to_categorical(np.array(labels), num_classes = self.num_classes)  
        
        return images, {'Embedding' : np.zeros_like(labels), 'Softmax' : labels}

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
    
    def load_dataset(self):
        for path in self.lines:
            path_split = path.split("\t")
            self.paths.append(path_split[0].split()[0])
            self.labels.append(int(path_split[1]))
        self.paths  = np.array(self.paths,dtype=np.object)
        self.labels = np.array(self.labels)
