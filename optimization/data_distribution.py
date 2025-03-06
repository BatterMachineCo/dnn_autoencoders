#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
File Name: data_distribution.py

Description: hyperparameter optimization

Author: junghwan lee
Email: jhrrlee@gmail.com
Date Created: 2023.10.14
Todo:
1. handling kernel crash
"""


# In[10]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
"""
cycle_sum: A 3-dimensional array representing the feature data. shape (no. of cells, no. of cycles, no. of features).
rul_repeated: A 2D array representing the target values. shape (no. of cells, no. of cycles).
eol: A 2D array representing the eol values used for stratification during the split.
training_size: A float value between 0 and 1.
bins: A list of bin edges used for digitizing the eol values to ensure stratified sampling.
"""
def data_distribution(cycle_sum, rul_repeated, eol, training_size, bins):
    X = cycle_sum
    y = rul_repeated
    indices = np.arange(len(X))
    random_state = 42  # Set the random state for reproducibility

    # Shuffle the indices
    np.random.seed(random_state)
    np.random.shuffle(indices)

    # Compute the class distribution based on y_bins
    y_bins = np.digitize(eol[:, 0], bins)
    class_distribution = np.bincount(y_bins)
    
    # Split the shuffled indices into training and validation sets with stratification
    train_size = training_size  # Proportion of data for training

    #random_state = None
    indices_train, indices_val = train_test_split(
        indices, test_size=1 - train_size, random_state=random_state, stratify=y_bins
    )
    
    # Split the data based on the obtained indices
    X_train, X_val = X[indices_train], X[indices_val]
    y_train, y_val = y[indices_train], y[indices_val]

    return X_train, X_val, y_train, y_val, indices_train, indices_val

def fit_scalers(X_train, y_train):
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler(feature_range=(0, 1))

    X_scaler.fit(X_train) # scalers per each feature
    y_scaler.fit(y_train)

    return X_scaler, y_scaler

def apply_normalization(X, y, X_scaler, y_scaler):
    X_norm = X_scaler.transform(X)
    y_norm = y_scaler.transform(y)

    return X_norm, y_norm
  
def save_scalers(X_scaler, y_scaler, X_scaler_filename='X_scaler.pkl', y_scaler_filename='y_scaler.pkl'):
    joblib.dump(X_scaler, X_scaler_filename)
    joblib.dump(y_scaler, y_scaler_filename)
    print(f"X_scaler saved as {X_scaler_filename}")
    print(f"y_scaler saved as {y_scaler_filename}")

def load_scalers(X_scaler_filename='X_scaler.pkl', y_scaler_filename='y_scaler.pkl'):
    X_scaler_loaded = joblib.load(X_scaler_filename)
    y_scaler_loaded = joblib.load(y_scaler_filename)
    return X_scaler_loaded, y_scaler_loaded

