import pandas as pd
import numpy as np
import tensorflow as tf
from utils import jaccard_discance
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate, add
from keras.optimizers import Adam
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import os

class UnetBrain():
    def __init__(self, image_size, learning_rate=0.00001, learning_decay=0.95, num_filters=16, kernel_size=3, pooling_size=2, dropout = 0.1, batch_norm=True):
        self.learning_rate = learning_rate
        self.learning_decay = learning_decay
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.pooling_size = pooling_size
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.image_size = image_size

        self.build_unet()

        self.configure_model()


    def conv2d_block(self, input_tensor, num_filters, kernel_size, batch_norm=True):
        # first layer
        x = Conv2D(filters = num_filters, kernel_size = (kernel_size, kernel_size), kernel_initializer='he_normal', padding='same')(input_tensor)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # second layer
        x = Conv2D(filters = num_filters, kernel_size = (kernel_size, kernel_size), kernel_initializer = 'he_normal', padding = 'same')(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x

    def build_unet(self):

        # input layer
        input_tensor = Input(shape =(self.image_size, self.image_size, 1))

        # encoder
        conv1 = self.conv2d_block(input_tensor,
                                  num_filters = self.num_filters*1,
                                  kernel_size = self.kernel_size,
                                  batch_norm = self.batch_norm)
        pool1 = MaxPooling2D(self.pooling_size, self.pooling_size)(conv1)
        pool1= Dropout(self.dropout)(pool1)

        conv2 = self.conv2d_block(pool1,
                                  num_filters = self.num_filters*2,
                                  kernel_size = self.kernel_size,
                                  batch_norm = self.batch_norm)
        pool2 = MaxPooling2D(self.pooling_size, self.pooling_size)(conv2)
        pool2= Dropout(self.dropout)(pool2)

        conv3 = self.conv2d_block(pool2,
                                  num_filters = self.num_filters*4,
                                  kernel_size = self.kernel_size,
                                  batch_norm = self.batch_norm)
        pool3 = MaxPooling2D(self.pooling_size, self.pooling_size)(conv3)
        pool3= Dropout(self.dropout)(pool3)

        conv4 = self.conv2d_block(pool3,
                                  num_filters = self.num_filters*8,
                                  kernel_size = self.kernel_size,
                                  batch_norm = self.batch_norm)
        pool4 = MaxPooling2D(self.pooling_size, self.pooling_size)(conv4)
        pool4= Dropout(self.dropout)(pool4)

        conv5 = self.conv2d_block(pool4,
                                  num_filters = self.num_filters*16,
                                  kernel_size = self.kernel_size,
                                  batch_norm = self.batch_norm)
        # decoder
        deconv6 = Conv2DTranspose(self.num_filters*8,
                                  kernel_size = (self.kernel_size, self.kernel_size),
                                  strides=(self.pooling_size, self.pooling_size),
                                  padding='same'
                                  )(conv5)
        deconv6 = concatenate([deconv6, conv4])
        deconv6 = Dropout(self.dropout)(deconv6)
        conv6 = self.conv2d_block(deconv6, self.num_filters*8, self.kernel_size, self.batch_norm)

        deconv7 = Conv2DTranspose(self.num_filters*4,
                                  kernel_size = (self.kernel_size, self.kernel_size),
                                  strides=(self.pooling_size, self.pooling_size),
                                  padding='same'
                                  )(conv6)
        deconv7 = concatenate([deconv7, conv3])
        deconv7 = Dropout(self.dropout)(deconv7)
        conv7 = self.conv2d_block(deconv7, self.num_filters*4, self.kernel_size, self.batch_norm)

        deconv8 = Conv2DTranspose(self.num_filters*2,
                                  kernel_size = (self.kernel_size, self.kernel_size),
                                  strides=(self.pooling_size, self.pooling_size),
                                  padding='same'
                                  )(conv7)
        deconv8 = concatenate([deconv8, conv2])
        deconv8 = Dropout(self.dropout)(deconv8)
        conv8 = self.conv2d_block(deconv8, self.num_filters*2, self.kernel_size, self.batch_norm)

        deconv9 = Conv2DTranspose(self.num_filters*1,
                                  kernel_size = (self.kernel_size, self.kernel_size),
                                  strides=(self.pooling_size, self.pooling_size),
                                  padding='same'
                                  )(conv8)
        deconv9 = concatenate([deconv9, conv1])
        deconv9 = Dropout(self.dropout)(deconv9)
        conv9 = self.conv2d_block(deconv9, self.num_filters*1, self.kernel_size, self.batch_norm)


        # output layer
        outputs = Conv2D(1, (1,1), activation='sigmoid')(conv9)
        # build functional model
        model = Model(inputs=[input_tensor], outputs=[outputs])
        self.model = model

    def configure_model(self):
        opt = Adam()
        self.model.compile(optimizer=opt, loss="binary_crossentropy", metrics=[jaccard_discance])
