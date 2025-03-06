"""
RUL_ML_Framework
File Name: cae_lib.py

Description: convolutional autoencoder

Author: junghwan lee

Email: jhrrlee@gmail.com
Date Created: 2023.09.24

"""

from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, AveragePooling1D, GlobalMaxPooling1D, Conv1DTranspose, GlobalAveragePooling1D
from tensorflow.keras.models import Model

# reduce time dimension, average pooling
def create_cae_v1(no_of_timeseries, no_of_features):
    input_shape = (no_of_timeseries, no_of_features)
    print("no_of_timeseries:", no_of_timeseries)
    print("no_of_features:", no_of_features)
    #input_shape = (no_of_features, no_of_timeseries)
    initializer = HeNormal(42)

    # Encoder
    inputs = Input(shape=input_shape)
    x = Conv1D(no_of_features, no_of_timeseries//100, activation='relu', padding='same', kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = AveragePooling1D(no_of_timeseries//100, padding='same')(x) # (8, 100)
    x = Conv1D(no_of_features, no_of_timeseries//100, activation='relu', padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(no_of_timeseries//100, padding='same')(x) # (8, 10)
    x = Conv1D(no_of_features, no_of_timeseries//100, activation='relu', padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(no_of_timeseries//100, padding='same')(x) # (8, 1)
    encoded = x

    # Decoder
    x = UpSampling1D(no_of_timeseries//100)(encoded)  # (10, 8)
    x = Conv1D(no_of_features, no_of_timeseries//100, activation='relu', padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(no_of_timeseries//100)(x)  # (100, 8)
    x = Conv1D(no_of_features, no_of_timeseries//100, activation='relu', padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(no_of_timeseries//100)(x)  # (1000, 8)
    x = Conv1D(no_of_features, no_of_timeseries//100, activation='relu', padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    decoded = x

    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    return autoencoder, encoder

# reduce time dimension, max pooling
def create_cae_v2(no_of_timeseries, no_of_features):
    input_shape = (no_of_timeseries, no_of_features)
    print("no_of_timeseries:", no_of_timeseries)
    print("no_of_features:", no_of_features)
    #input_shape = (no_of_features, no_of_timeseries)
    initializer = HeNormal(42)

    # Encoder
    inputs = Input(shape=input_shape)
    x = Conv1D(no_of_features, no_of_timeseries//100, activation='relu', padding='same', kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(no_of_timeseries//100, padding='same')(x) # (8, 100)
    x = Conv1D(no_of_features, no_of_timeseries//100, activation='relu', padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(no_of_timeseries//100, padding='same')(x) # (8, 10)
    x = Conv1D(no_of_features, no_of_timeseries//100, activation='relu', padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(no_of_timeseries//100, padding='same')(x) # (8, 1)
    encoded = x

    # Decoder
    x = UpSampling1D(no_of_timeseries//100)(encoded)  # (10, 8)
    x = Conv1D(no_of_features, no_of_timeseries//100, activation='relu', padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(no_of_timeseries//100)(x)  # (100, 8)
    x = Conv1D(no_of_features, no_of_timeseries//100, activation='relu', padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(no_of_timeseries//100)(x)  # (1000, 8)
    x = Conv1D(no_of_features, no_of_timeseries//100, activation='relu', padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    decoded = x

    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    return autoencoder, encoder


def create_cae_v3(no_of_timeseries, no_of_features):
    input_shape = (no_of_timeseries, no_of_features)
    print("no_of_timeseries:", no_of_timeseries)
    print("no_of_features:", no_of_features)
    #input_shape = (no_of_features, no_of_timeseries)
    initializer = HeNormal(42)

    # Encoder
    inputs = Input(shape=input_shape)
    x = Conv1D(128, no_of_timeseries//100, activation='relu', padding='same', kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = AveragePooling1D(no_of_timeseries//100, padding='same')(x) # (128, 10)
    x = Conv1D(64, no_of_timeseries//100, activation='relu', padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(no_of_timeseries//100, padding='same')(x) # (64, 10)
    x = Conv1D(32, no_of_timeseries//100, activation='relu', padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(no_of_timeseries//100, padding='same')(x) # (32, 10)
    encoded = Conv1D(16, 1, activation='linear', padding='same', kernel_initializer=initializer)(x)  # (1, 1)

    # Decoder
    x = UpSampling1D(no_of_timeseries//100)(encoded)  # # (32, 10)
    x = Conv1D(32, no_of_timeseries//100, activation='relu', padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(no_of_timeseries//100)(x)  # (64, 10)
    x = Conv1D(64, no_of_timeseries//100, activation='relu', padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(no_of_timeseries//100)(x)  # (128, 10)
    x = Conv1D(128, no_of_timeseries//100, activation='relu', padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    
    decoded = Conv1D(no_of_features, 1, activation='linear', padding='same', kernel_initializer=initializer)(x)  # (1000,8)
    #decoded = x

    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    return autoencoder, encoder



