#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
File Name: AnomalyDetector.ipynb

Description: anamaly detect

Author: junghwan lee
Email: jhrrlee@gmail.com
Date Created: 2023.09.12
Todo:
1. Implement a function to resume training in case of kernel crash or stops
"""


# **Objective:**  
# To develop an anomaly detection mechanism using an autoencoder trained with K-fold cross-validation. The process aims to compute error metrics on a per-cell basis, average them across K-folds, and identify anomalies based on these metrics.
# 
# **Method:**  
# The outlined process consists of the following steps:
# 
# 1. **Training Phase:**
#     - The dataset is divided into K folds.
#     - An Autoencoder is trained K times, once for each fold, resulting in K trained Autoencoder models.
#     - In each iteration, one fold is used for validation while the rest are used for training.
# 
# 2. **Error Calculation Phase:**
#     - Compute error metrics (MAPE, RMSE, MAE, MSE) for each cell in the validation set of each fold.
#     - These error metrics are computed between the original and the reconstructed data obtained from the Autoencoder.
#     - These computations result in K sets of error metrics, one for each fold.
# 
# 3. **Averaging Phase:**
#     - Average the error metrics computed in the previous phase across all K folds.
#     - This results in a single set of average error metrics for each cell.
# 
# 4. **Anomaly Detection Phase:**
#     - Detect anomalies based on the specified error metric and a provided or computed threshold.
#     - Flag the cells whose error metrics exceed the threshold as anomalies.
# 
# 5. **Visualization Phase:**
#     - Visualize the average error metrics by cell using bar charts, one for each error metric (MAPE, RMSE, MAE, MSE).
#     - Highlight the anomalies in red on these charts to clearly indicate the anomalous cells.
# 
# **Functions:**
# 
# - `train_autoencoder(data, no_of_folds)`:  
#   Train the Autoencoder using K-fold cross-validation.
# 
# - `compute_reconstruction_error(model)`:  
#   Compute the reconstruction error metrics (MAPE, RMSE, MAE, MSE) for each cell.
# 
# - `detect_anomalies(average_error_metrics_by_cells, metric, threshold_function, threshold_value)`:  
#   Detect anomalies based on a specified error metric and threshold.
# 
# - `plot_average_error_metrics_by_cells(average_error_metrics_by_cells, anomalies)`:  
#   Visualize the average error metrics by cell, highlighting the anomalies.
# 

# In[17]:
import time 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tqdm.keras import TqdmCallback
#20230928
import os
import pandas as pd
from openpyxl import load_workbook
from adjustText import adjust_text
# Abstract ExcelWriter class
class ExcelWriter:
    def write(self, file_path, df, sheet_name):
        raise NotImplementedError("This method should be overridden by subclass")

# OpenpyxlWriter class for 'openpyxl' engine
class OpenpyxlWriter(ExcelWriter):
    def write(self, file_path, df, sheet_name):
        if os.path.exists(file_path):
            # If the file exists, load the workbook and append to it
            book = load_workbook(file_path)
            with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            # If the file does not exist, create a new one
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

# XlsxWriter class for 'xlsxwriter' engine
class XlsxWriter(ExcelWriter):
    def write(self, file_path, df, sheet_name):
        # Always create a new file with 'xlsxwriter' engine
        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

# Factory to return the correct writer based on the engine
class ExcelWriterFactory:
    @staticmethod
    def get_writer(engine):
        writers = {
            'openpyxl': OpenpyxlWriter(),
            'xlsxwriter': XlsxWriter()
        }
        writer = writers.get(engine)
        if writer is None:
            raise ValueError(f"Unsupported engine: {engine}")
        return writer

# Function to write DataFrame to Excel
def write_to_excel(file_path, df, sheet_name, engine='openpyxl'):
    # Use the factory to get the correct writer based on the engine
    writer = ExcelWriterFactory.get_writer(engine)
    writer.write(file_path, df, sheet_name)


# In[25]:


import tensorflow as tf
def mape(y_true, y_pred):
    epsilon = 1e-9  # Adding a small constant to avoid division by zero
    error = (y_true - y_pred) / (tf.abs(y_true) + epsilon)
    return 100.0 * tf.reduce_mean(tf.abs(error))

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


# In[18]:


#20230928
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ReconErrCalc:
    def __init__(self, data, reconstructed_data):
        self.true_data = data.get_data()  # Assuming BatData has a method get_data() that returns the data array
        self.reconstructed_data = reconstructed_data  # assuming the data is already unnormalized
        self.no_of_cells, self.no_of_cycles, self.no_of_timeseries, self.no_of_features = self.true_data.shape

    def _compute_error_metrics(self, true_data, reconstructed_data):
        mse_val = mean_squared_error(true_data, reconstructed_data)
        rmse_val = np.sqrt(mse_val)
        mae_val = mean_absolute_error(true_data, reconstructed_data)
        mape_val = mape(true_data, reconstructed_data)  # Assuming mape is a custom defined function or imported
        return [mse_val, rmse_val, mae_val, mape_val]

    def _compute_error(self, true_data, reconstructed_data, axis):
        per_error_metrics = np.empty((true_data.shape[axis], 4))  # 4 for mse, rmse, mae, mape
        for i in range(true_data.shape[axis]):
            indices = [slice(None)] * len(true_data.shape)
            indices[axis] = i
            true_slice = true_data[tuple(indices)].flatten()
            reconstructed_slice = reconstructed_data[tuple(indices)].flatten()
            per_error_metrics[i] = self._compute_error_metrics(true_slice, reconstructed_slice)
        return per_error_metrics

    def compute_error_by_cells(self):
        return self._compute_error(self.true_data, self.reconstructed_data, 0)

    def compute_error_by_features(self):
        return self._compute_error(self.true_data, self.reconstructed_data, 3)

    def compute_error_by_cycles(self):
        return self._compute_error(self.true_data, self.reconstructed_data, 1)

    def compute_error_by_cells_cycles(self):
        per_cell_cycle_error_metrics = np.empty((self.no_of_cells, self.no_of_cycles, 4))  # 4 for mse, rmse, mae, mape
        for i in range(self.no_of_cells):
            for j in range(self.no_of_cycles):
                cell_cycle_true_data = self.true_data[i, j, :, :].flatten()
                cell_cycle_reconstructed_data = self.reconstructed_data[i, j, :, :].flatten()
                per_cell_cycle_error_metrics[i, j] = self._compute_error_metrics(cell_cycle_true_data, cell_cycle_reconstructed_data)
        return per_cell_cycle_error_metrics


# In[19]:


class BatData:
    def __init__(self, data):
        self.data = data #self.no_of_cells, self.no_of_cycles, self.no_of_timeseries, self.no_of_features
        self.normalized_data, self.scalers = self.normalize_data_byfeatures(data)

    def normalize_data_byfeatures(self, data):
        # data shape: (no_of_cells, no_of_cycles, no_of_timeseries, no_of_features)
        assert len(data.shape) == 4, "Expected data to have 4 dimensions."
        reshaped_data = data.reshape(-1, data.shape[-1])
        scalers = [StandardScaler() for _ in range(data.shape[-1])]

        # Normalize each feature separately
        normalized_data = np.empty_like(reshaped_data, dtype=float)
        for i, scaler in enumerate(scalers):
            normalized_data[:, i] = scaler.fit_transform(reshaped_data[:, i].reshape(-1, 1)).flatten()

        return normalized_data.reshape(data.shape), scalers

    def unnormalize_data_byfeatures(self, normalized_data):
        reshaped_normalized_data = normalized_data.reshape(-1, normalized_data.shape[-1])
        unnormalized_data = np.empty_like(reshaped_normalized_data, dtype=float)
        for i, scaler in enumerate(self.scalers):
            unnormalized_data[:, i] = scaler.inverse_transform(reshaped_normalized_data[:, i].reshape(-1, 1)).flatten()
        unnormalized_data = unnormalized_data.reshape(normalized_data.shape)
        return unnormalized_data
    def get_data(self):
      return self.data
    def get_normalized_data(self):
      return self.normalized_data
    def get_scalers(self):
      return self.scalers
    def get_shape(self):
      return self.data.shape


# In[20]:


import traceback
# Define error methods mapping
ERROR_METHODS = {
    "cell": "compute_error_by_cells",
    "features": "compute_error_by_features",
    "cycles": "compute_error_by_cycles",
    "cell_cycle": "compute_error_by_cells_cycles"
}

class kFoldAutoencoderTrainer: #unique name for a test
    def __init__(self, base_dir, test_name, data,  model, n_splits=5, model_dir=None, evaluation_dir=None):
        self.test_name = test_name
        self.data = data
        self.autoencoder = model
        self.initial_weights = model.get_weights()
        self.n_splits = n_splits
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.model_dir = model_dir or os.path.join(base_dir, test_name, 'model')
        self.evaluation_dir = evaluation_dir or os.path.join(base_dir, test_name, 'evaluation')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.evaluation_dir, exist_ok=True)

    def train(self, learning_rate = 0.001, batch_size=100, max_epoch=100, no_patience=10):
        self.history = {}

        history_file_path = os.path.join(self.evaluation_dir, f'{self.test_name}_history.xlsx')

        for fold_num, (train_idx, val_idx) in enumerate(self.kfold.split(self.data.get_normalized_data())):
            train_data = self.data.get_normalized_data()[train_idx]
            val_data = self.data.get_normalized_data()[val_idx]

            self.autoencoder.set_weights(self.initial_weights)

            optimizer = Adam(learning_rate=learning_rate, epsilon=1e-09)
            self.autoencoder.compile(optimizer=optimizer, loss='mse', metrics=[rmse, 'mae', mape])

            # Train the autoencoder
            print(train_data.shape)
            train_data = train_data.reshape(-1, train_data.shape[2], train_data.shape[3])
            val_data = val_data.reshape(-1, val_data.shape[2], val_data.shape[3])
            history = self.train_autoencoder(self.autoencoder, train_data, val_data, fold_num, batch_size, max_epoch, no_patience)
            self.history[f'fold_{fold_num + 1}'] = history.history  # Save training history

            print(f'complete training_fold_{fold_num + 1}')

            # Convert data to DataFrames
            history_df = pd.DataFrame(history.history)

            write_to_excel(history_file_path, history_df, sheet_name=f'Fold{fold_num + 1}')

    def train_autoencoder(self, autoencoder, train_data, val_data, fold_num, batch_size, max_epoch, no_patience):
        early_stopping = EarlyStopping(monitor='val_loss', patience=no_patience, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(
          #os.path.join(self.model_dir, f'best_model_fold_{fold_num}.h5'),
          os.path.join(self.model_dir, f'best_model_fold_{fold_num}.keras'),
          monitor='val_loss',
          save_best_only=True,
          save_weights_only=False
        )
        try:
            history = autoencoder.fit(
                train_data, train_data,
                epochs=max_epoch,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(val_data, val_data),
                callbacks=[early_stopping, model_checkpoint, TqdmCallback(verbose=1)],
                verbose=0
            )
            return history
        except Exception as e:
            print(f"An error occurred while training the autoencoder: {e}")
            traceback.print_exc()
            return None  # Return None if an error occurs

    def prediction(self, fold_num):
        custom_objects = {'rmse': rmse, 'mape': mape}
        #model_path = os.path.join(self.model_dir, f'best_model_fold_{fold_num}.h5')
        model_path = os.path.join(self.model_dir, f'best_model_fold_{fold_num}.keras')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model file not found: {model_path}')
        autoencoder = load_model(model_path, custom_objects=custom_objects)
        norm_data = self.data.get_normalized_data()
        norm_data = norm_data.reshape(-1, norm_data.shape[2], norm_data.shape[3])
        return autoencoder.predict(norm_data)

    def evaluate(self, error_type="cell"):
        all_folds_error_metrics = []  # List to hold per-cell error metrics for each fold
        error_metrics_file_path = os.path.join(self.evaluation_dir, f'{self.test_name}_{error_type}_error_metrics.xlsx')

        for fold_num in range(self.n_splits):
            reconstructed_data_norm  = self.prediction(fold_num)

            reconstructed_data = self.data.unnormalize_data_byfeatures(reconstructed_data_norm)
            reconstructed_data = reconstructed_data.reshape(self.data.get_shape())
            err_calc = ReconErrCalc(self.data, reconstructed_data)
            per_cell_error_metrics = err_calc.compute_error_by_cells()  # base evaluation
            all_folds_error_metrics.append(per_cell_error_metrics)
            error_metrics_df = pd.DataFrame(per_cell_error_metrics, columns=['MSE', 'RMSE', 'MAE', 'MAPE'])
            write_to_excel(error_metrics_file_path, error_metrics_df, sheet_name=f'Fold{fold_num + 1}')

        all_folds_error_metrics_array = np.array(all_folds_error_metrics)
        average_error_metrics_by_cells = np.mean(all_folds_error_metrics_array, axis=0)

        return average_error_metrics_by_cells

class AutoencoderTrainer:
    def __init__(self, base_dir, test_name, data, model, model_dir=None, evaluation_dir=None):
        self.test_name = test_name
        self.data = data
        self.autoencoder = model
        self.model_dir = model_dir or os.path.join(base_dir, test_name, 'model')
        self.evaluation_dir = evaluation_dir or os.path.join(base_dir, test_name, 'evaluation')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.evaluation_dir, exist_ok=True)

    def train(self, learning_rate=0.001, batch_size=100, max_epoch=100, no_patience=10):
        train_data = self.data.get_normalized_data()

        # Create the optimizer and compile the autoencoder
        optimizer = Adam(learning_rate=learning_rate, epsilon=1e-09)

        self.autoencoder.compile(optimizer=optimizer, loss='mse', metrics=[rmse, 'mae', mape])

        train_data = train_data.reshape(-1, train_data.shape[2], train_data.shape[3])

        history = self.train_autoencoder(self.autoencoder, train_data, batch_size, max_epoch, no_patience)

        # Saving the training history
        history_df = pd.DataFrame(history.history)
        history_file_path = os.path.join(self.evaluation_dir, f'{self.test_name}_history.xlsx')
        write_to_excel(history_file_path, history_df, sheet_name=f'{self.test_name}')

    def train_autoencoder(self, autoencoder, train_data, batch_size, max_epoch, no_patience):
        early_stopping = EarlyStopping(monitor='loss', patience=no_patience, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(
            #os.path.join(self.model_dir, 'best_model.h5'),
            os.path.join(self.model_dir, 'best_model.keras'),
            monitor='loss',
            save_best_only=True,
            save_weights_only=False
        )
        try:
            history = autoencoder.fit(
                train_data, train_data,
                epochs=max_epoch,
                batch_size=batch_size,
                shuffle=True,
                callbacks=[early_stopping, model_checkpoint, TqdmCallback(verbose=1)],
                verbose=0
            )
            return history
        except Exception as e:
            print(f"An error occurred while training the autoencoder: {e}")
            traceback.print_exc()
            return None

    # def prediction(self):
    #     custom_objects = {'rmse': rmse, 'mape': mape}
    #     #model_path = os.path.join(self.model_dir, 'best_model.h5')
    #     model_path = os.path.join(self.model_dir, 'best_model.keras')
    #     if not os.path.exists(model_path):
    #         raise FileNotFoundError(f'Model file not found: {model_path}')
    #     autoencoder = load_model(model_path, custom_objects=custom_objects)
    #     norm_data = self.data.get_normalized_data()
    #     norm_data = norm_data.reshape(-1, norm_data.shape[2], norm_data.shape[3])
    #     return autoencoder.predict(norm_data)

    def prediction(self):
        custom_objects = {'rmse': rmse, 'mape': mape}
        # model_path = os.path.join(self.model_dir, 'best_model.h5')
        model_path = os.path.join(self.model_dir, 'best_model.keras')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model file not found: {model_path}')
        
        autoencoder = load_model(model_path, custom_objects=custom_objects)
        norm_data = self.data.get_normalized_data()
        norm_data = norm_data.reshape(-1, norm_data.shape[2], norm_data.shape[3])
        
        start_time = time.time()
        predictions = autoencoder.predict(norm_data)
        end_time = time.time()
        
        print(f"Prediction time: {end_time - start_time} seconds")
        return predictions

    def evaluate(self, error_type="cell"):
        reconstructed_data_norm = self.prediction()
        reconstructed_data = self.data.unnormalize_data_byfeatures(reconstructed_data_norm)
        reconstructed_data = reconstructed_data.reshape(self.data.get_shape())
        err_calc = ReconErrCalc(self.data, reconstructed_data)
        error_metrics = err_calc.compute_error_by_cells()

        error_metrics_df = pd.DataFrame(error_metrics, columns=['MSE', 'RMSE', 'MAE', 'MAPE'])
        error_metrics_file_path = os.path.join(self.evaluation_dir, f'{self.test_name}_{error_type}_error_metrics.xlsx')
        write_to_excel(error_metrics_file_path, error_metrics_df, sheet_name=f'{self.test_name}')

        return error_metrics



def threshold_std(errors, sigma=3):
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    threshold = mean_error + sigma * std_error
    return threshold

def threshold_percentile(errors, percentile=95):
    threshold = np.percentile(errors, percentile)
    return threshold
  
class AnomalyDetector:
    def __init__(self):
        self.test = 1
        
    def detect_anomalies(self, average_error_metrics_by_cells, metric='mse', threshold_function=None, threshold_value=None):
        metric_dict = {'mse': 0, 'rmse': 1, 'mae': 2, 'mape': 3}
        metric_idx = metric_dict.get(metric, 0)

        # Get the specified metric errors for all cells
        metric_errors = average_error_metrics_by_cells[:, metric_idx]

        # If a threshold value is provided, use it as the threshold
        if threshold_function is not None:
            anomaly_threshold = threshold_function(metric_errors, threshold_value)
        # Otherwise, default to computing the threshold based on the mean and standard deviation of the errors
        else:
            anomaly_threshold = np.mean(metric_errors) + 2 * np.std(metric_errors)

        # Identify the indices of the cells that have errors exceeding the threshold
        anomalies = np.where(metric_errors > anomaly_threshold)[0]

        return anomalies, metric_errors

    # def visual(self, average_error_metrics_by_cells, anomalies):
    #     # Ensure data is a numpy array for consistency
    #     average_error_metrics_by_cells = np.array(average_error_metrics_by_cells)

    #     # Assume the metrics names are in the order: MSE, RMSE, MAE, MAPE
    #     metrics_names = ['MSE', 'RMSE', 'MAE', 'MAPE']
    #     cell_indices = np.arange(average_error_metrics_by_cells.shape[0])  # Create an array of cell indices

    #     # Create a new figure
    #     plt.figure(figsize=(20, 20))

    #     # Plotting each metric
    #     for i, metric_name in enumerate(metrics_names):
    #         plt.subplot(4, 1, i + 1)  # Create a subplot for each metric

    #         # Scatter plot for all data points
    #         plt.scatter(cell_indices, average_error_metrics_by_cells[:, i], label=f'{metric_name}')

    #         # Overlay scatter plot for anomalies in red
    #         plt.scatter(anomalies, average_error_metrics_by_cells[anomalies, i], color='red', label=f'{metric_name} Anomalies' if i==0 else "", zorder=5)

    #         plt.title(f'Average {metric_name} by Cell Across Folds')
    #         plt.xlabel('Cell Index')
    #         plt.ylabel(f'Average {metric_name}')
    #         plt.xticks(cell_indices, rotation=90)  # Show all cell indices on the x-axis with rotation for better visibility

    #         # Add legend
    #         if i == 0:
    #             plt.legend()

    #     plt.tight_layout()  # Adjusts the layout so that plots do not overlap
    #     plt.show()
    # def visual(self, average_error_metrics_by_cells, anomalies):
    #     metrics_names = ['MSE', 'MAE', 'MAPE', 'RMSE']
    #     cell_indices = np.arange(average_error_metrics_by_cells.shape[0])

    #     for i, metric_name in enumerate(metrics_names):
    #         plt.figure(figsize=(10, 5), dpi=300)  # High-quality figure for journals

    #         # Plot normal cells
    #         plt.scatter(cell_indices, average_error_metrics_by_cells[:, i], 
    #                     label='Normal Cells', alpha=0.7)

    #         # Highlight anomalous cells distinctly
    #         plt.scatter(anomalies, average_error_metrics_by_cells[anomalies, i],
    #                     color='red', label='Anomalous Cells', edgecolors='black', linewidth=0.5, s=80, zorder=5)

    #         plt.xlabel('Cell Index', fontsize=20)
    #         plt.ylabel(f'Average {metric_name}', fontsize=20)
    #         plt.xticks(fontsize=20)
    #         plt.yticks(fontsize=20)
    #         plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
    #         plt.legend(fontsize=20)
    #         #plt.grid(alpha=0.5, linestyle='--')
    #         plt.tight_layout()
    #         plt.show()
    
    # def plot_abnormal_cells(autoencoder, X_train_reshaped, percentile):
    # X_pred = autoencoder.predict(X_train_reshaped)
    # print(X_train_reshaped.shape)
    # # Calculate RMSE for each cycle
    # errors = np.sqrt(np.mean(np.square(X_train_reshaped - X_pred), axis=(1, 2)))
    # print(X_train_reshaped.shape[0])
    # # Reshape errors back to original shape
    # errors = errors.reshape(117, int(X_train_reshaped.shape[0]/117))

    # # Calculate the threshold based on the percentile
    # threshold = np.percentile(errors, percentile)

    # # Calculate the maximum loss per cell
    # max_loss_per_cell = np.max(errors, axis=1)

    # # Find the indices of abnormal cells
    # abnormal_cell_indices = np.where(max_loss_per_cell > threshold)[0]

    # # Plot the loss of individual cells
    # plt.figure(figsize=(12, 6))

    # # Plot normal cells in blue
    # plt.scatter(range(117), max_loss_per_cell, color='b', label='Normal Cells')

    # # Plot abnormal cells in red
    # plt.scatter(abnormal_cell_indices, max_loss_per_cell[abnormal_cell_indices], color='r', label='Abnormal Cells')

    # # Add text annotations for abnormal cells
    # for i in abnormal_cell_indices:
    #     plt.text(i, max_loss_per_cell[i], str(i), color='black', fontsize=8, ha='center', va='bottom', weight='bold')

    # plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    # plt.xlabel('Cell Index')
    # plt.ylabel('Loss')
    # plt.title('Loss of Individual Cells')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    def visual(self, average_error_metrics_by_cells, anomalies, percentile=80, anomaly_size=20, anomaly_offset=5, legend_visible=False):
        metrics_names = ['MSE', 'MAE', 'MAPE', 'RMSE']
        cell_indices = np.arange(average_error_metrics_by_cells.shape[0])

        for i, metric_name in enumerate(metrics_names):
            plt.figure(figsize=(10, 5), dpi=300)  # High-quality figure for journals

            # Compute the threshold based on the given percentile for this metric
            threshold = np.percentile(average_error_metrics_by_cells[:, i], percentile)

            # Plot all cells (normal cells)
            plt.scatter(cell_indices, average_error_metrics_by_cells[:, i],
                        label='Normal Cells', alpha=0.7)

            # Highlight anomalous cells distinctly
            plt.scatter(anomalies, average_error_metrics_by_cells[anomalies, i],
                        color='red', label='Anomalous Cells', edgecolors='black', 
                        linewidth=0.5, s=80, zorder=5)

            # Draw a horizontal line to indicate the threshold
            plt.axhline(threshold, color='red', linestyle='--', 
                        label=f'Threshold ({percentile}th percentile)')

# Create a list to store text objects
            texts = []
            for anomaly in anomalies:
                txt = plt.text(anomaly, average_error_metrics_by_cells[anomaly, i],
                            str(anomaly), color='black', fontsize=anomaly_size, 
                            ha='center', va='bottom', weight='bold', zorder=10)
                texts.append(txt)

            # Adjust text positions to reduce overlap
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray'))

            plt.xlabel('Cell Index', fontsize=20)
            plt.ylabel(f'Average {metric_name}', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

            # Update the title to include the count of anomalies
            #plt.title(f'Average {metric_name} (Anomalies: {len(anomalies)})', fontsize=20)
            if legend_visible:
                plt.legend(fontsize=20)
            plt.tight_layout()
            plt.show()

