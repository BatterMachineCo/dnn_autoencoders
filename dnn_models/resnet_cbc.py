#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
File Name: resnet_cbc.py

Description: resnet_cbc

Author: junghwan lee
Email: jhrrlee@gmail.com
Date Created: 2023.10.14

"""

from tensorflow.keras.layers import Dense, Conv1D, Flatten, Input, BatchNormalization, Activation, Add, Dropout, MaxPooling1D, GlobalAveragePooling1D
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
from keras import backend as K

# def basic_block(input_tensor, filters, kernel_size, activation, strides=1, use_bn=True):
#     x = Conv1D(filters, kernel_size, strides=strides, padding='same', kernel_initializer='he_normal')(input_tensor)
#     if use_bn:
#         x = BatchNormalization()(x)
#     x = Activation(activation)(x)

#     x = Conv1D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
#     if use_bn:
#         x = BatchNormalization()(x)

#     if strides != 1 or K.int_shape(input_tensor)[-1] != filters:
#         shortcut = Conv1D(filters, 1, strides=strides, padding='same', kernel_initializer='he_normal')(input_tensor)
#         if use_bn:
#             shortcut = BatchNormalization()(shortcut)
#     else:
#         shortcut = input_tensor

#     x = Add()([x, shortcut])
#     x = Activation(activation)(x)
#     return x

def basic_block(input_tensor, filters, kernel_size, activation, strides=1, use_bn=True):
    x = Conv1D(filters, kernel_size, strides=strides, padding='same', kernel_initializer='he_normal')(input_tensor)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)

    x = Conv1D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    if use_bn:
        x = BatchNormalization()(x)

    if strides != 1 or input_tensor.shape[-1] != filters:  # Fixed line
        shortcut = Conv1D(filters, 1, strides=strides, padding='same', kernel_initializer='he_normal')(input_tensor)
        if use_bn:
            shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = input_tensor

    x = Add()([x, shortcut])
    x = Activation(activation)(x)
    return x


def identity_block(input_tensor, filters, kernel_size=3, activation='relu', use_bn=True):
    f1, f2, f3 = filters

    # First layer
    x = Conv1D(f1, 1, padding='same', kernel_initializer='he_normal')(input_tensor)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)

    # Second layer
    x = Conv1D(f2, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)

    # Third layer
    x = Conv1D(f3, 1, padding='same', kernel_initializer='he_normal')(x)
    if use_bn:
        x = BatchNormalization()(x)

    if input_tensor.shape[-1] != f3:
        input_tensor = Conv1D(f3, 1, padding='same', kernel_initializer='he_normal')(input_tensor)
        if use_bn:
            input_tensor = BatchNormalization()(input_tensor)

    x = Add()([x, input_tensor])
    x = Activation(activation)(x)
    return x

def conv_block(input_tensor, filters, kernel_size=3, strides=1, activation='relu', use_bn=True):
    f1, f2, f3 = filters

    # First layer
    x = Conv1D(f1, 1, strides=strides, padding='same', kernel_initializer='he_normal')(input_tensor)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)

    # Second layer
    x = Conv1D(f2, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)

    # Third layer
    x = Conv1D(f3, 1, padding='same', kernel_initializer='he_normal')(x)
    if use_bn:
        x = BatchNormalization()(x)

    if input_tensor.shape[-1] != f3 or strides != 1:
        shortcut = Conv1D(f3, 1, strides=strides, padding='same', kernel_initializer='he_normal')(input_tensor)
        if use_bn:
            shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = input_tensor

    x = Add()([x, shortcut])
    x = Activation(activation)(x)
    return x


def gen_resnet18_model(input_shape, kernel_size=3, strides=2, activation='relu', learning_rate=0.0001, loss='mse', metrics=['mean_absolute_error'], use_bn=True):
    model_input = Input(shape=input_shape)

    # Initial Convolution
    x = Conv1D(64, 7, strides=2, padding='same', activation=None, kernel_initializer='he_normal')(model_input)  # Remove the activation here
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    x = MaxPooling1D(3, strides=2, padding='same')(x)

    # Stage 2
    x = basic_block(x, 64, kernel_size, activation, use_bn=use_bn)
    x = basic_block(x, 64, kernel_size, activation, use_bn=use_bn)

    # Stage 3
    x = basic_block(x, 128, kernel_size, activation, strides, use_bn=use_bn)
    x = basic_block(x, 128, kernel_size, activation, use_bn=use_bn)

    # Stage 4
    x = basic_block(x, 256, kernel_size, activation, strides, use_bn=use_bn)
    x = basic_block(x, 256, kernel_size, activation, use_bn=use_bn)

    # Stage 5
    x = basic_block(x, 512, kernel_size, activation, strides, use_bn=use_bn)
    x = basic_block(x, 512, kernel_size, activation, use_bn=use_bn)

    x = GlobalAveragePooling1D()(x)
    output = Dense(1, name='rul')(x)

    model = Model(inputs=model_input, outputs=output)
    optimizer = Adam(learning_rate=learning_rate, epsilon=1e-07)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

def gen_resnet34_model(input_shape, kernel_size=3, strides=2, activation='relu', learning_rate=0.0001, loss='mse', metrics=['mean_absolute_error'], use_bn=True):
    model_input = Input(shape=input_shape)

    # Initial Convolution
    x = Conv1D(64, 7, strides=2, padding='same', activation=None, kernel_initializer='he_normal')(model_input)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    x = MaxPooling1D(3, strides=2, padding='same')(x)

    # Stage 2
    for _ in range(3):
        x = basic_block(x, 64, kernel_size, activation, use_bn=use_bn)

    # Stage 3
    x = basic_block(x, 128, kernel_size, activation, strides, use_bn=use_bn)
    for _ in range(3):
        x = basic_block(x, 128, kernel_size, activation, use_bn=use_bn)

    # Stage 4
    x = basic_block(x, 256, kernel_size, activation, strides, use_bn=use_bn)
    for _ in range(5):
        x = basic_block(x, 256, kernel_size, activation, use_bn=use_bn)

    # Stage 5
    x = basic_block(x, 512, kernel_size, activation, strides, use_bn=use_bn)
    for _ in range(2):
        x = basic_block(x, 512, kernel_size, activation, use_bn=use_bn)

    x = GlobalAveragePooling1D()(x)
    output = Dense(1, name='rul')(x)

    model = Model(inputs=model_input, outputs=output)
    optimizer = Adam(learning_rate=learning_rate, epsilon=1e-07)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

  
def gen_resnet50_model(input_shape, kernel_size=3, strides=2, activation='relu', learning_rate=0.0001, loss='mse', metrics=['mean_absolute_error'], use_bn=True):
    model_input = Input(shape=input_shape)

    # Stage 1
    x = Conv1D(64, 7, strides=2, padding='same', kernel_initializer='he_normal')(model_input)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = MaxPooling1D(3, strides=2, padding='same')(x)

    # Stage 2
    x = conv_block(x, filters=(64, 64, 256), kernel_size=kernel_size, activation=activation, use_bn=use_bn)
    for _ in range(2):
        x = identity_block(x, filters=(64, 64, 256), kernel_size=kernel_size, activation=activation, use_bn=use_bn)

    # Stage 3
    x = conv_block(x, filters=(128, 128, 512), kernel_size=kernel_size, strides=strides, activation=activation, use_bn=use_bn)
    for _ in range(3):
        x = identity_block(x, filters=(128, 128, 512), kernel_size=kernel_size, activation=activation, use_bn=use_bn)

    # Stage 4
    x = conv_block(x, filters=(256, 256, 1024), kernel_size=kernel_size, strides=strides, activation=activation, use_bn=use_bn)
    for _ in range(5):
        x = identity_block(x, filters=(256, 256, 1024), kernel_size=kernel_size, activation=activation, use_bn=use_bn)

    # Stage 5
    x = conv_block(x, filters=(512, 512, 2048), kernel_size=kernel_size, strides=strides, activation=activation, use_bn=use_bn)
    for _ in range(2):
        x = identity_block(x, filters=(512, 512, 2048), kernel_size=kernel_size, activation=activation, use_bn=use_bn)

    x = GlobalAveragePooling1D()(x)
    output = Dense(1, name='rul')(x)

    model = Model(inputs=model_input, outputs=output)
    optimizer = Adam(learning_rate=learning_rate, epsilon=1e-07)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
  
def gen_resnet101_model(input_shape, kernel_size=3, strides=2, activation='relu', learning_rate=0.0001, loss='mse', metrics=['mean_absolute_error'], use_bn=True):
    model_input = Input(shape=input_shape)

    # Stage 1
    x = Conv1D(64, 7, strides=2, padding='same', kernel_initializer='he_normal')(model_input)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = MaxPooling1D(3, strides=2, padding='same')(x)

    # Stage 2
    x = conv_block(x, filters=(64, 64, 256), kernel_size=kernel_size, activation=activation, use_bn=use_bn)
    for _ in range(2):
        x = identity_block(x, filters=(64, 64, 256), kernel_size=kernel_size, activation=activation, use_bn=use_bn)

    # Stage 3
    x = conv_block(x, filters=(128, 128, 512), kernel_size=kernel_size, strides=strides, activation=activation, use_bn=use_bn)
    for _ in range(3):
        x = identity_block(x, filters=(128, 128, 512), kernel_size=kernel_size, activation=activation, use_bn=use_bn)

    # Stage 4
    x = conv_block(x, filters=(256, 256, 1024), kernel_size=kernel_size, strides=strides, activation=activation, use_bn=use_bn)
    for _ in range(22):
        x = identity_block(x, filters=(256, 256, 1024), kernel_size=kernel_size, activation=activation, use_bn=use_bn)

    # Stage 5
    x = conv_block(x, filters=(512, 512, 2048), kernel_size=kernel_size, strides=strides, activation=activation, use_bn=use_bn)
    for _ in range(2):
        x = identity_block(x, filters=(512, 512, 2048), kernel_size=kernel_size, activation=activation, use_bn=use_bn)

    x = GlobalAveragePooling1D()(x)
    output = Dense(1, name='rul')(x)

    model = Model(inputs=model_input, outputs=output)
    optimizer = Adam(learning_rate=learning_rate, epsilon=1e-07)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def gen_resnet152_model(input_shape, kernel_size=3, strides=2, activation='relu', learning_rate=0.0001, loss='mse', metrics=['mean_absolute_error'], use_bn=True):
    model_input = Input(shape=input_shape)

    # Stage 1
    x = Conv1D(64, 7, strides=2, padding='same', kernel_initializer='he_normal')(model_input)
    if use_bn:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = MaxPooling1D(3, strides=2, padding='same')(x)

    # Stage 2
    x = conv_block(x, filters=(64, 64, 256), kernel_size=kernel_size, activation=activation, use_bn=use_bn)
    for _ in range(2):
        x = identity_block(x, filters=(64, 64, 256), kernel_size=kernel_size, activation=activation, use_bn=use_bn)

    # Stage 3
    x = conv_block(x, filters=(128, 128, 512), kernel_size=kernel_size, strides=strides, activation=activation, use_bn=use_bn)
    for _ in range(7):
        x = identity_block(x, filters=(128, 128, 512), kernel_size=kernel_size, activation=activation, use_bn=use_bn)

    # Stage 4
    x = conv_block(x, filters=(256, 256, 1024), kernel_size=kernel_size, strides=strides, activation=activation, use_bn=use_bn)
    for _ in range(35):
        x = identity_block(x, filters=(256, 256, 1024), kernel_size=kernel_size, activation=activation, use_bn=use_bn)

    # Stage 5
    x = conv_block(x, filters=(512, 512, 2048), kernel_size=kernel_size, strides=strides, activation=activation, use_bn=use_bn)
    for _ in range(2):
        x = identity_block(x, filters=(512, 512, 2048), kernel_size=kernel_size, activation=activation, use_bn=use_bn)

    x = GlobalAveragePooling1D()(x)
    output = Dense(1, name='rul')(x)

    model = Model(inputs=model_input, outputs=output)
    optimizer = Adam(learning_rate=learning_rate, epsilon=1e-07)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

