#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
File Name: hype_optimizer_cbc.py

Description: hyperparameter optimization

Author: junghwan lee
Email: jhrrlee@gmail.com
Date Created: 2023.10.09

Modifications:
2023.10.29: 
  - Added support for GridSampler in Optuna. (by junghwan lee)
  - to maintain previous version experiments, new class is made.

Todo:
1. handling kernel crash
"""


# In[1]:


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xlsxwriter
from tqdm import tqdm
import optuna
import pandas as pd
import time
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sqlite3
from tqdm.keras import TqdmCallback
# issue 2:
from keras.callbacks import ModelCheckpoint, EarlyStopping


# In[2]:


def root_mean_squared_error(y_true, y_pred):
    epsilon = 1e-6  # Small value to avoid sqrt(0)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true) + epsilon))


def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = 1e-2  # A slightly larger small constant to prevent extreme values
    y_true = tf.maximum(y_true, epsilon)  # Ensures y_true is not too small
    return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true))



# In[4]:


from tensorflow.keras.callbacks import Callback
from tqdm.notebook import tqdm  # for Jupyter notebooks

class TQDMProgressBar(Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.tqdm_bar = tqdm(total=self.epochs, position=0, desc='Training Progress')

    def on_epoch_end(self, epoch, logs=None):
        self.tqdm_bar.update(1)

    def on_train_end(self, logs=None):
        self.tqdm_bar.close()


# In[5]:


def set_trial_to_running(db_name, study_name, trial_numbers):
    # Construct the path to the SQLite database
    db_path = os.path.join("db", db_name)
    #db_path = f'sqlite:///db\\{db_name}'

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get the study_id from the study_name
    cursor.execute("SELECT study_id FROM studies WHERE study_name=?", (study_name,))
    study_id = cursor.fetchone()
    if not study_id:
        print(f"Study '{study_name}' not found.")
        return
    study_id = study_id[0]

    # Set the state of specific trials to RUNNING
    for trial_number in trial_numbers:
        try:
            cursor.execute("UPDATE trials SET state=? WHERE study_id=? AND trial_id=?", (optuna.trial.TrialState.RUNNING.value, study_id, trial_number))
            print(f"Trial {trial_number} set to RUNNING.")
        except Exception as e:
            print(f"Error updating trial {trial_number}: {e}")

    # Commit the changes and close the connection
    conn.commit()
    conn.close()


# In[6]:


def set_trial_to_running_directly(db_path, db_name, study_name, trial_numbers):
    # Construct the full path to the SQLite database with the .sqlite3 extension
    full_db_path = os.path.join(db_path, db_name)

    # Connect to the SQLite database
    conn = sqlite3.connect(full_db_path)
    cursor = conn.cursor()

    # Get the study_id corresponding to the study_name
    cursor.execute("SELECT study_id FROM studies WHERE study_name=?", (study_name,))
    study_id = cursor.fetchone()
    if not study_id:
        print(f"No study found with the name: {study_name}")
        return
    study_id = study_id[0]

    # Change the state of specific trials to RUNNING
    for trial_number in trial_numbers:
        cursor.execute("UPDATE trials SET state='RUNNING' WHERE study_id=? AND trial_id=?", (study_id, trial_number))
        if cursor.rowcount:
            print(f"Trial {trial_number} state set to RUNNING.")
        else:
            print(f"Trial {trial_number} not found for study {study_name}.")

    # Commit the changes and close the connection
    conn.commit()
    conn.close()


# In[7]:


# class model_epoch_save(Callback):
#     def __init__(self, model, model_name, epoch_file_name, trial, monitor='val_loss'):
#         super().__init__()
#         self.monitor = monitor
#         self.best_val_loss = float('inf')
#         self.epoch_file_name = epoch_file_name
#         self.model_name = model_name
#         self.trial = trial
#         self.model = model

#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         val_loss = logs.get(self.monitor)
#         print("epoch end", self.best_val_loss, val_loss)
#         if val_loss < self.best_val_loss:
#             self.best_val_loss = val_loss

#             # Save model
#             self.model.save(self.model_name)
#             # Save epoch number
#             with open(self.epoch_file_name, 'w') as f:
#                 f.write(str(epoch))
#             print("model is saved", self.model_name, self.epoch_file_name)
#         else:
#             print("model is not saved")

from tensorflow.keras.callbacks import Callback

class model_epoch_save(Callback):
    def __init__(self, model, model_name, epoch_file_name, trial, monitor='val_loss'):
        super().__init__()
        self.monitor = monitor
        self.best_val_loss = float('inf')
        self.epoch_file_name = epoch_file_name
        self.model_name = model_name
        self.trial = trial
        self._model_instance = model  # Rename to avoid conflicts

    @property
    def model(self):
        """Override Keras' model property to avoid conflicts."""
        return self._model_instance

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get(self.monitor, None)

        if val_loss is None:
            print(f"Warning: {self.monitor} not found in logs. Skipping save.")
            return
        
        print(f"Epoch {epoch} End | Best Val Loss: {self.best_val_loss}, Current Val Loss: {val_loss}")

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

            # Save model
            self.model.save(self.model_name)
            # Save epoch number
            with open(self.epoch_file_name, 'w') as f:
                f.write(str(epoch))

            print(f"✅ Model saved: {self.model_name}, Epoch file: {self.epoch_file_name}")
        else:
            print("❌ Model not saved (no improvement).")
            
def read_epoch_value(file_name):
    with open(file_name, 'r') as f:
        epoch = int(f.read().strip())
    return epoch

# issue 2:
from keras.callbacks import ModelCheckpoint

def train_model(model,
                model_name,
                custom_checkpoint,
                batch_size,
                X_train_norm,
                y_train_norm,
                X_val_norm,
                y_val_norm,
                start_epoch = 0,
                end_epoch=1000,
                patience = 100
                ):

    start_time = time.time()

    # issue 2: there are performance issues might be caused by custom_checkpoint or TqdmCallback.
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=patience)
    checkpoint = ModelCheckpoint(model_name, monitor='val_loss', save_best_only=True)

    history = model.fit(
        X_train_norm, y_train_norm,
        validation_data=(X_val_norm, y_val_norm),
        epochs=end_epoch,
        initial_epoch=start_epoch,
        batch_size=batch_size,
        callbacks=[es, checkpoint, TqdmCallback(verbose=0)],
        verbose=0
    )

    training_time = time.time() - start_time

    return history, training_time

def load_best_model(model_name, custom_objects):
    if os.path.exists(model_name):
        return load_model(model_name, custom_objects=custom_objects)
    elif os.path.exists(final_model_name):
        return load_model(final_model_name, custom_objects=custom_objects)
    else:
        return None

class FilenameGenerator:
    def __init__(self, study_name, base_path):
        self.study_name = study_name

        # Define subdirectories within the study directory
        self.model_dir = os.path.join(base_path, "models/")
        self.results_dir = os.path.join(base_path, "results/")
        self.epoch_info_dir = os.path.join(base_path, "epoch_info/")

        # Create directories if they don't exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.epoch_info_dir, exist_ok=True)

    def model_name(self, trial_number):
        return f"{self.model_dir}{self.study_name}_{trial_number}.keras"

    def results_name(self, trial_number):
        return f"{self.results_dir}{self.study_name}_results_{trial_number}.xlsx"

    def epoch_file_name(self, trial_number):
        return f"{self.epoch_info_dir}epoch_info_{trial_number}.txt"

def evaluate_model(model, X_train_norm, y_train_norm, X_val_norm, y_val_norm, y_scaler, y_train, y_val):
    # Start the timer
    start_time = time.time()

    # Get model's predictions on the training set
    y_train_pred_norm = model.predict(X_train_norm)
    # Unnormalize the predictions
    y_train_pred = y_scaler.inverse_transform(y_train_pred_norm).flatten()
    eva_y_train = np.squeeze(y_train)
    # Compute evaluation metrics for training set
    train_rmse_rul = np.sqrt(mean_squared_error(eva_y_train, y_train_pred))
    train_mae_rul = mean_absolute_error(eva_y_train, y_train_pred)
    train_mape_rul = mean_absolute_percentage_error(eva_y_train, y_train_pred).numpy().item()

    # Print training set evaluation metrics
    print("Training set metrics:")
    print("RMSE:", train_rmse_rul)
    print("MAE:", train_mae_rul)
    print("MAPE:", train_mape_rul)

    # Get model's predictions on the validation set
    y_val_pred_norm = model.predict(X_val_norm)
    # Unnormalize the predictions
    y_val_pred = y_scaler.inverse_transform(y_val_pred_norm).flatten()
    eva_y_val = np.squeeze(y_val)
    # Compute evaluation metrics for validation set
    val_rmse_rul = np.sqrt(mean_squared_error(eva_y_val, y_val_pred))
    val_mae_rul = mean_absolute_error(eva_y_val, y_val_pred)
    val_mape_rul = mean_absolute_percentage_error(eva_y_val, y_val_pred).numpy().item()

    # Print validation set evaluation metrics
    print("Validation set metrics:")
    print("RMSE:", val_rmse_rul)
    print("MAE:", val_mae_rul)
    print("MAPE:", val_mape_rul)

    # Calculate the prediction time
    prediction_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    return {
        "train_rmse_rul": train_rmse_rul,
        "train_mae_rul": train_mae_rul,
        "train_mape_rul": train_mape_rul,
        "val_rmse_rul": val_rmse_rul,
        "val_mae_rul": val_mae_rul,
        "val_mape_rul": val_mape_rul,
        "prediction_time": prediction_time,
        "train_pred": y_train_pred,
        "val_pred": y_val_pred,
        "train_y": eva_y_train,
        "val_y": eva_y_val
    }

def set_trial_attributes(trial, metrics, training_time):
    trial.set_user_attr('training_time', training_time)
    trial.set_user_attr('prediction_time', metrics["prediction_time"])
    trial.set_user_attr('train_mape_rul', metrics["train_mape_rul"])
    trial.set_user_attr('train_rmse_rul', metrics["train_rmse_rul"])
    trial.set_user_attr('train_mae_rul', metrics["train_mae_rul"])
    trial.set_user_attr('val_mape_rul', metrics["val_mape_rul"])
    trial.set_user_attr('val_rmse_rul', metrics["val_rmse_rul"])
    trial.set_user_attr('val_mae_rul', metrics["val_mae_rul"])

def save_results(filename, history, trial, eva_y_train, train_results, eva_y_val, val_results):
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        # Save the history data to its own sheet
        history_df = pd.DataFrame(history.history)
        history_df.to_excel(writer, sheet_name="History", index=False)

        # Create separate DataFrames for train and validation datasets
        train_data = pd.DataFrame({'y_train': eva_y_train, 'y_train_pred': train_results})
        val_data = pd.DataFrame({'y_val': eva_y_val, 'y_val_pred': val_results})

        # Save each DataFrame to a different sheet in the same Excel file
        train_data.to_excel(writer, sheet_name='Train', index=False)
        val_data.to_excel(writer, sheet_name='Validation', index=False)

def prediction_with_norm(model, X_train_norm, y_train_norm, X_val_norm, y_val_norm, y_scaler, y_train, y_val):
    # Start the timer
    start_time = time.time()

    # Get model's predictions on the training set
    y_train_pred_norm = model.predict(X_train_norm)
    # Unnormalize the predictions
    y_train_pred = y_scaler.inverse_transform(y_train_pred_norm).flatten()
    eva_y_train = np.squeeze(y_train)
    # Compute evaluation metrics for training set
    train_rmse_rul = np.sqrt(mean_squared_error(eva_y_train, y_train_pred))
    train_mae_rul = mean_absolute_error(eva_y_train, y_train_pred)
    train_mape_rul = mean_absolute_percentage_error(eva_y_train, y_train_pred).numpy().item()

    # Print training set evaluation metrics
    print("Training set metrics:")
    print("RMSE:", train_rmse_rul)
    print("MAE:", train_mae_rul)
    print("MAPE:", train_mape_rul)

    # Get model's predictions on the validation set
    y_val_pred_norm = model.predict(X_val_norm)
    # Unnormalize the predictions
    y_val_pred = y_scaler.inverse_transform(y_val_pred_norm).flatten()
    eva_y_val = np.squeeze(y_val)
    # Compute evaluation metrics for validation set
    val_rmse_rul = np.sqrt(mean_squared_error(eva_y_val, y_val_pred))
    val_mae_rul = mean_absolute_error(eva_y_val, y_val_pred)
    val_mape_rul = mean_absolute_percentage_error(eva_y_val, y_val_pred).numpy().item()

    # Print validation set evaluation metrics
    print("Validation set metrics:")
    print("RMSE:", val_rmse_rul)
    print("MAE:", val_mae_rul)
    print("MAPE:", val_mape_rul)

    # Calculate the prediction time
    prediction_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    return {
        "train_rmse_rul": train_rmse_rul,
        "train_mae_rul": train_mae_rul,
        "train_mape_rul": train_mape_rul,
        "val_rmse_rul": val_rmse_rul,
        "val_mae_rul": val_mae_rul,
        "val_mape_rul": val_mape_rul,
        "prediction_time": prediction_time,
        "train_pred": y_train_pred,
        "val_pred": y_val_pred,
        "train_y": eva_y_train,
        "val_y": eva_y_val
    }


# In[8]:


class HypeOptimizer:
    def __init__(self, hype_space, custom_objects, n_trial, base_path, db_name, study_name, direction='minimize'):
        self.init_model = hype_space
        self.n_trial = n_trial
        self.custom_objects = custom_objects

        study_path = os.path.join(base_path, study_name)
        db_path = os.path.join(base_path, 'db')

        if not os.path.exists(study_path):
            os.makedirs(study_path)
            print(f"Study Folder '{study_path}' created!")
        else:
            print(f"Study Folder '{study_path}' already exists!")

        if not os.path.exists(db_path):
            os.makedirs(db_path)
            print(f"DB Folder '{db_path}' created!")
        else:
            print(f"DB Folder '{db_path}' already exists!")

        #Initialize an Optuna study with the SQLite database.
        self.study = optuna.create_study(storage=f'sqlite:///{os.path.join(db_path, db_name)}',
                                         study_name=study_name,
                                         direction=direction,
                                         load_if_exists=True)

        self.filenames = FilenameGenerator(study_name=study_name, base_path=study_path)

    # ND, future feature for handling kernel crash.
    # easy to remove RUNNING state, but no way to directly change trials in optuna DB.
    def run(self, X_train_norm, y_train_norm, X_val_norm, y_val_norm, y_scaler, y_train, y_val, max_epoch=100, patience=100):
        # Call the function to remove trials with the "RUNNING" state
        # remove_running_trials_from_db(db_path, db_name)
        # remove_running_trials_and_reset_count(db_path, db_name, study_name)

        # ND, is there a better way?
        # Pass the necessary parameters to the objective function using the unpacking operator
        func = lambda trial: self.objective(trial,
                                            X_train_norm,
                                            y_train_norm,
                                            X_val_norm,
                                            y_val_norm,
                                            y_scaler,
                                            y_train,
                                            y_val,
                                            max_epoch,
                                            patience=100)
        self.study.optimize(func, n_trials=self.n_trial)

        df = self.study.trials_dataframe(attrs=('number', 'value', 'params', 'state', 'user_attrs'))

        # Renaming 'value' to 'MAPE' for clarity and 'user_attrs_prediction_time' to 'Prediction Time (ms)'
        df.rename(columns={'value': 'MAPE',
                           'user_attrs_prediction_time': 'Prediction Time (ms)'}, inplace=True)

        print("Best Hyperparameters:", self.study.best_params)
        print("Best score:", self.study.best_value)

        return df

    def objective(self, trial, X_train_norm, y_train_norm, X_val_norm, y_val_norm, y_scaler, y_train, y_val, max_epoch, patience=100):
        input_shape = X_train_norm.shape[1:]
        previous_epoch = 0
        epoch_file_name = self.filenames.epoch_file_name(trial.number)
        model_name = self.filenames.model_name(trial.number)
        savefile_name = self.filenames.results_name(trial.number)

        K.clear_session()
        tf.compat.v1.reset_default_graph()

        print('traial: ', trial.number)
        print('epoch_file_name: ', epoch_file_name)
        print('model_name: ', model_name)
        print('savefile_name: ', savefile_name)

        # when kernel crash. but this will not work due to trial numbers in db file.
        if os.path.exists(epoch_file_name):
            # If an epoch file exists, load the saved model
            previous_epoch = read_epoch_value(epoch_file_name)
            print(f'Loading model from:{model_name}, Previous epoch: {previous_epoch}')
            model = load_model(model_name, custom_objects=self.custom_objects)
            # here need to read the last epoch number which will become start epoch when kernel crash.
        else:
            # Initialize a new model if training has not been started
            print("Initializing a new model.")
            model, batch_size = self.init_model(trial, input_shape)

        print("trial:", trial, epoch_file_name)

        custom_checkpoint = model_epoch_save(model, model_name, epoch_file_name, trial, monitor='val_loss')

        #issue 2
        history, training_time = train_model(model,
                                            model_name,
                                            custom_checkpoint,
                                            batch_size,
                                            X_train_norm,
                                            y_train_norm,
                                            X_val_norm,
                                            y_val_norm,
                                            start_epoch = previous_epoch,
                                            end_epoch = max_epoch,
                                            patience = patience)

        # Delete epoch file after successful trial completion
        if os.path.exists(epoch_file_name):
            print("Deleting epoch file:", epoch_file_name)
            os.remove(epoch_file_name)

        if history is None:
            print("Training did not complete successfully.")

        model = load_model(model_name, custom_objects=self.custom_objects)
        if model is None:
            print("Best model not found.")
            mape_score = float('inf')
        else:
            metrics = evaluate_model(model, X_train_norm, y_train_norm, X_val_norm, y_val_norm, y_scaler, y_train, y_val)
            set_trial_attributes(trial, metrics, training_time)

            # Save the results
            save_results(savefile_name,
                         history,
                         trial,
                         metrics["train_y"],
                         metrics["train_pred"],
                         metrics["val_y"],
                         metrics["val_pred"])
            mape_score = (metrics["train_mape_rul"] + metrics["val_mape_rul"]) / 2

        return mape_score

    def get_trial_results(self, trial_number=None):
        df = self.study.trials_dataframe(attrs=('number', 'value', 'params', 'state', 'user_attrs'))

        if trial_number is not None:
            df = df[df['number'] == trial_number]

        df.rename(columns={'value': 'MAPE',
                          'user_attrs_prediction_time': 'Prediction Time (ms)'}, inplace=True)
        return df

    def get_trained_model(self, trial_number):
        model_filename = self.filenames.model_name(trial_number)

        if not os.path.exists(model_filename):
            print(f"Model for trial {trial_number} not found.")
            return None

        model = load_model(model_filename, custom_objects=self.custom_objects)

        return model


# In[ ]:


class HypeOptimizer_v2:
    def __init__(self, sampler, hype_space, custom_objects, n_trial, base_path, db_name, study_name, direction='minimize'):
        self.init_model = hype_space
        self.n_trial = n_trial
        self.custom_objects = custom_objects

        study_path = os.path.join(base_path, study_name)
        db_path = os.path.join(base_path, 'db')

        if not os.path.exists(study_path):
            os.makedirs(study_path)
            print(f"Study Folder '{study_path}' created!")
        else:
            print(f"Study Folder '{study_path}' already exists!")

        if not os.path.exists(db_path):
            os.makedirs(db_path)
            print(f"DB Folder '{db_path}' created!")
        else:
            print(f"DB Folder '{db_path}' already exists!")

        #Initialize an Optuna study with the SQLite database.
        self.study = optuna.create_study(storage=f'sqlite:///{os.path.join(db_path, db_name)}',
                                         sampler=sampler,
                                         study_name=study_name,
                                         direction=direction,
                                         load_if_exists=True)

        self.filenames = FilenameGenerator(study_name=study_name, base_path=study_path)

    # ND, future feature for handling kernel crash.
    # easy to remove RUNNING state, but no way to directly change trials in optuna DB.
    def run(self, X_train_norm, y_train_norm, X_val_norm, y_val_norm, y_scaler, y_train, y_val, max_epoch=100, patience=100):
        # Call the function to remove trials with the "RUNNING" state
        # remove_running_trials_from_db(db_path, db_name)
        # remove_running_trials_and_reset_count(db_path, db_name, study_name)

        # ND, is there a better way?
        # Pass the necessary parameters to the objective function using the unpacking operator
        func = lambda trial: self.objective(trial,
                                            X_train_norm,
                                            y_train_norm,
                                            X_val_norm,
                                            y_val_norm,
                                            y_scaler,
                                            y_train,
                                            y_val,
                                            max_epoch,
                                            patience=patience)
        self.study.optimize(func, n_trials=self.n_trial)

        df = self.study.trials_dataframe(attrs=('number', 'value', 'params', 'state', 'user_attrs'))

        # Renaming 'value' to 'MAPE' for clarity and 'user_attrs_prediction_time' to 'Prediction Time (ms)'
        df.rename(columns={'value': 'MAPE',
                           'user_attrs_prediction_time': 'Prediction Time (ms)'}, inplace=True)

        print("Best Hyperparameters:", self.study.best_params)
        print("Best score:", self.study.best_value)

        return df

    def objective(self, trial, X_train_norm, y_train_norm, X_val_norm, y_val_norm, y_scaler, y_train, y_val, max_epoch, patience=100):
        input_shape = X_train_norm.shape[1:]
        previous_epoch = 0
        epoch_file_name = self.filenames.epoch_file_name(trial.number)
        model_name = self.filenames.model_name(trial.number)
        savefile_name = self.filenames.results_name(trial.number)

        print('traial: ', trial.number)
        print('epoch_file_name: ', epoch_file_name)
        print('model_name: ', model_name)
        print('savefile_name: ', savefile_name)

        # 🛑 Fix: Check if model and epoch files exist before loading
        if os.path.exists(epoch_file_name) and os.path.exists(model_name):
            try:
                previous_epoch = read_epoch_value(epoch_file_name)
                print(f'🔄 Resuming training from {model_name}, Previous epoch: {previous_epoch}')

                model = load_model(model_name, custom_objects=self.custom_objects)
                print(f"✅ Model loaded successfully: {model_name}")

            except Exception as e:
                print(f"⚠️ Error loading model: {e}. Reinitializing a new model.")
                model, batch_size = self.init_model(trial, input_shape)
                previous_epoch = 0  # Restart training from epoch 0
        else:
            print("🚀 No previous model found. Initializing a new model.")
            model, batch_size = self.init_model(trial, input_shape)
            previous_epoch = 0  # Start from epoch 0


        print("trial:", trial, epoch_file_name)

        custom_checkpoint = model_epoch_save(model, model_name, epoch_file_name, trial, monitor='val_loss')

        #issue 2
        history, training_time = train_model(model,
                                            model_name,
                                            custom_checkpoint,
                                            batch_size,
                                            X_train_norm,
                                            y_train_norm,
                                            X_val_norm,
                                            y_val_norm,
                                            start_epoch = previous_epoch,
                                            end_epoch = max_epoch,
                                            patience = patience)

        # Delete epoch file after successful trial completion
        if os.path.exists(epoch_file_name):
            print("Deleting epoch file:", epoch_file_name)
            os.remove(epoch_file_name)

        if history is None:
            print("Training did not complete successfully.")

        if os.path.exists(model_name):
            try:
                model = load_model(model_name, custom_objects=self.custom_objects)
            except Exception as e:
                model = None

        if model is None:
            print("Best model not found.")
            mape_score = float('inf')
        else:
            metrics = evaluate_model(model, X_train_norm, y_train_norm, X_val_norm, y_val_norm, y_scaler, y_train, y_val)
            set_trial_attributes(trial, metrics, training_time)

            # Save the results
            save_results(savefile_name,
                         history,
                         trial,
                         metrics["train_y"],
                         metrics["train_pred"],
                         metrics["val_y"],
                         metrics["val_pred"])
            mape_score = (metrics["train_mape_rul"] + metrics["val_mape_rul"]) / 2

        K.clear_session()
        tf.compat.v1.reset_default_graph()

        return mape_score

    def get_trial_results(self, trial_number=None):
        df = self.study.trials_dataframe(attrs=('number', 'value', 'params', 'state', 'user_attrs'))

        if trial_number is not None:
            df = df[df['number'] == trial_number]

        df.rename(columns={'value': 'MAPE',
                          'user_attrs_prediction_time': 'Prediction Time (ms)'}, inplace=True)
        return df

    def get_trained_model(self, trial_number):
        model_filename = self.filenames.model_name(trial_number)

        if not os.path.exists(model_filename):
            print(f"Model for trial {trial_number} not found.")
            return None

        model = load_model(model_filename, custom_objects=self.custom_objects)

        return model


# In[9]:


def visualization(model, X_train, y_train, X_val, y_val, battery_ids_train, battery_ids_val):
    # Initialize scaler for input features and target values
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler(feature_range=(0, 1))

    X_train_norm = X_scaler.fit_transform(X_train)
    X_val_norm = X_scaler.transform(X_val)
    y_train_norm = y_scaler.fit_transform(y_train)
    y_val_norm = y_scaler.transform(y_val)
    y_train = np.squeeze(y_train)
    y_val = np.squeeze(y_val)

    # Get model's predictions on the validation set
    y_pred_norm = model.predict(X_train_norm)

    # Unnormalize the predictions
    y_pred = y_scaler.inverse_transform(y_pred_norm).flatten()

    # Compute evaluation metrics on the unnormalized predictions
    rmse_rul = np.mean(np.sqrt(mean_squared_error(y_train, y_pred)))
    mape_rul = mean_absolute_percentage_error(y_train, y_pred)
    print(f"RMSE for RUL: {rmse_rul} MAPE for RUL: {mape_rul}")

    # Calculate the number of batteries and cycles
    n_batteries = len(battery_ids_train)
    n_cycles = X_train.shape[0] // n_batteries

    # Create an expanded version of battery_ids_train where each ID is repeated for each cycle
    expanded_battery_ids_train = np.repeat(battery_ids_train, n_cycles)

    # Find the indices of the first cycle for each cell using np.where
    first_cycle_indices = np.where(np.diff(expanded_battery_ids_train) != 0)[0] + 1

    # Scatter plot of true and predicted values
    plt.figure(figsize=(14, 6))
    plt.scatter(np.arange(len(y_train)), y_train, label='True')
    plt.scatter(np.arange(len(y_train)), y_pred, label='Predicted')

    # Plot cell IDs for first cycle points
    plt.scatter(first_cycle_indices, y_train[first_cycle_indices], c='red', label='Cell ID')
    for i, idx in enumerate(first_cycle_indices):
        plt.text(idx, y_train[idx], expanded_battery_ids_train[idx], ha='center', va='bottom')

    plt.xlabel('Data Point (Cell Number * Cycle Number)')
    plt.ylabel('RUL')
    plt.title('Scatter Plot of True and Predicted RUL (Train Set)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # Scatter plot of difference (Train Set)
    diff = y_train - y_pred
    plt.figure(figsize=(12, 7))
    plt.scatter(np.arange(len(y_train)), diff, color='r', label='Difference')
    plt.xlabel('Data Point (Cell Number * Cycle Number)')
    plt.ylabel('Difference')
    plt.title('Scatter Plot of Difference Between True and Predicted RUL (Train Set)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Get model's predictions on the validation set
    y_pred_norm = model.predict(X_val_norm)

    # Unnormalize the predictions
    y_pred = y_scaler.inverse_transform(y_pred_norm).flatten()

# Compute evaluation metrics on the unnormalized predictions
    rmse_rul = np.sqrt(mean_squared_error(y_val, y_pred))
    mape_rul = mean_absolute_percentage_error(y_val, y_pred)
    print(f"RMSE for RUL: {rmse_rul} MAPE for RUL: {mape_rul}")

    # Calculate the number of batteries and cycles
    n_batteries_val = len(battery_ids_val)
    n_cycles_val = X_val.shape[0] // n_batteries_val

    # Create an expanded version of battery_ids_val where each ID is repeated for each cycle
    expanded_battery_ids_val = np.repeat(battery_ids_val, n_cycles_val)

    # Find the indices of the first cycle for each cell using np.where
    first_cycle_indices_val = np.where(np.diff(expanded_battery_ids_val) != 0)[0] + 1

    # Scatter plot of true and predicted values
    plt.figure(figsize=(12, 6))
    plt.scatter(np.arange(len(y_val)), y_val, label='True')
    plt.scatter(np.arange(len(y_val)), y_pred, label='Predicted')

    # Plot cell IDs for first cycle points
    plt.scatter(first_cycle_indices_val, y_val[first_cycle_indices_val], c='red', label='Cell ID')
    for i, idx in enumerate(first_cycle_indices_val):
        plt.text(idx, y_val[idx], expanded_battery_ids_val[idx], ha='center', va='bottom')

    plt.xlabel('Data Point (Cell Number * Cycle Number)')
    plt.ylabel('RUL')
    plt.legend(loc='upper left')
    plt.title('Scatter Plot of True and Predicted RUL (Validation Set)')
    plt.tight_layout()
    plt.show()

    # Scatter plot of difference
    diff_val = y_val - y_pred
    plt.figure(figsize=(12, 6))
    plt.scatter(np.arange(len(y_val)), diff_val, color='r', label='Difference')
    plt.xlabel('Data Point (Cell Number * Cycle Number)')
    plt.ylabel('Difference')
    plt.title('Scatter Plot of Difference Between True and Predicted RUL (Validation Set)')
    plt.tight_layout()
    plt.show()

    return rmse_rul

