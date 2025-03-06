"""
File Name: seq2seq_ae.ipynb

Description: seq2seq auto encoder

Author: junghwan lee
Email: jhrrlee@gmail.com
Date Created: 2023.09.12

"""

from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, AdditiveAttention
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal

def create_seq2seq_ae_v1(no_of_timeseries, no_of_features):
    input_shape = (no_of_timeseries, no_of_features)
    initializer = HeNormal(42)

    # Encoder
    encoder_inputs = Input(shape=input_shape)
    print(encoder_inputs.shape)
    encoded = LSTM(no_of_features, return_sequences=False, kernel_initializer=initializer)(encoder_inputs)

    # Decoder
    decoder_inputs = RepeatVector(no_of_timeseries)(encoded)
    print(decoder_inputs.shape)
    decoded = LSTM(no_of_features, return_sequences=True, kernel_initializer=initializer)(decoder_inputs)

    # Autoencoder
    autoencoder = Model(encoder_inputs, decoded)
    encoder = Model(encoder_inputs, encoded)

    return autoencoder, encoder

from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Reshape

def create_seq2seq_ae_v2(no_of_timeseries, no_of_features):
    input_shape = (no_of_timeseries, no_of_features)
    initializer = HeNormal(42)

    # Determine the chunk size and number of chunks
    num_chunks = 10
    chunk_size = no_of_timeseries // num_chunks

    # Encoder
    encoder_inputs = Input(shape=input_shape)
    x = Reshape((num_chunks, chunk_size, no_of_features))(encoder_inputs)
    encoded_chunks = TimeDistributed(LSTM(no_of_features, return_sequences=False, kernel_initializer=initializer))(x)
    # expected dimension of encoded_chunks (num_chunks, no_of_features)

    # Decoder
    x = TimeDistributed(RepeatVector(chunk_size))(encoded_chunks)
    x = Reshape((num_chunks, chunk_size, no_of_features))(x)

    decoded_chunks = TimeDistributed(LSTM(no_of_features, return_sequences=True, kernel_initializer=initializer))(x)

    # Reshape the decoded chunks back to the original input shape
    decoded = Reshape((no_of_timeseries, no_of_features))(decoded_chunks)

    # Autoencoder
    autoencoder = Model(encoder_inputs, decoded)

    # Encoder Model (Optional, in case you need it later)
    encoder = Model(encoder_inputs, encoded_chunks)

    return autoencoder, encoder


def create_seq2seq_ae_v3(no_of_timeseries, no_of_features):
    input_shape = (no_of_timeseries, no_of_features)
    initializer = HeNormal(42)

    # Determine the chunk size and number of chunks
    num_chunks = 10
    chunk_size = no_of_timeseries // num_chunks

    # Encoder
    encoder_inputs = Input(shape=input_shape)
    x = Reshape((num_chunks, chunk_size, no_of_features))(encoder_inputs)
    x = TimeDistributed(LSTM(no_of_features, return_sequences=False, kernel_initializer=initializer))(x) # (10, 8)
    encoded = LSTM(no_of_features, return_sequences=False, kernel_initializer=initializer)(x) # (1, 8)
    # expected dimension of encoded_chunks (num_chunks, no_of_features)

    # Decoder
    x = RepeatVector(num_chunks)(encoded) # (10, 8)
    x = LSTM(no_of_features, return_sequences=True, kernel_initializer=initializer)(x) # (10, 8)
    x = TimeDistributed(RepeatVector(chunk_size))(x) # (1000, 8)
    x = Reshape((num_chunks, chunk_size, no_of_features))(x) # (10, 100, 8)
    decoded_chunks = TimeDistributed(LSTM(no_of_features, return_sequences=True, kernel_initializer=initializer))(x)

    # Reshape the decoded chunks back to the original input shape
    decoded = Reshape((no_of_timeseries, no_of_features))(decoded_chunks)

    # Autoencoder
    autoencoder = Model(encoder_inputs, decoded)

    # Encoder Model (Optional, in case you need it later)
    encoder = Model(encoder_inputs, encoded)

    return autoencoder, encoder

