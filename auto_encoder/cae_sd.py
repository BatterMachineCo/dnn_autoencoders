#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
RUL_ML_Framework
File Name: cae_sd.py

Description: convolutional autoencoder reducing spatial dimension for statistical features.

Author: junghwan lee

Email: jhrrlee@gmail.com
Date Created: 2023.10.16

"""


from keras.layers import Input, Conv1D, BatchNormalization, UpSampling1D, Flatten, Dense, Reshape
from keras.models import Model
from tensorflow.keras.initializers import HeNormal



def create_cae_sd(no_of_features, fixed_channel_dim=1, use_bn=True, kernel_size=3, no_of_filters=128):
    input_shape = (no_of_features, fixed_channel_dim)
    initializer = HeNormal(seed=42)

    # Encoder
    inputs = Input(shape=input_shape)

    x = Conv1D(no_of_filters, kernel_size, activation='relu', padding='same', strides=2, kernel_initializer=initializer)(inputs)
    if use_bn:
        x = BatchNormalization()(x)
    x = Conv1D(no_of_filters//2, kernel_size, activation='relu', padding='same', strides=2, kernel_initializer=initializer)(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = Conv1D(no_of_filters//4, kernel_size, activation='relu', padding='same', strides=2, kernel_initializer=initializer)(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = Conv1D(no_of_filters//8, kernel_size, activation='relu', padding='same', strides=1, kernel_initializer=initializer)(x)
    if use_bn:
        x = BatchNormalization()(x)

    encoded = Conv1D(1, 1, activation='linear', padding='same', kernel_initializer=initializer)(x)

    # Decoder
    x = UpSampling1D(2)(encoded)  # double the feature dimension
    x = Conv1D(no_of_filters//8, kernel_size, activation='relu', padding='same', kernel_initializer=initializer)(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)  # double the feature dimension
    x = Conv1D(no_of_filters//4, kernel_size, activation='relu', padding='same', kernel_initializer=initializer)(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)  # double the feature dimension
    x = Conv1D(no_of_filters//2, kernel_size, activation='relu', padding='same', kernel_initializer=initializer)(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = Conv1D(no_of_filters, kernel_size, activation='relu', padding='same', kernel_initializer=initializer)(x)
    x = Flatten()(x)
    decoded = Dense(no_of_features * fixed_channel_dim, activation='linear')(x)
    decoded = Reshape((no_of_features, fixed_channel_dim))(decoded)

    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    return autoencoder, encoder




