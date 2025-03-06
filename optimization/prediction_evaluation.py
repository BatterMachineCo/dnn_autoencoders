#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
File Name: prediction_evaluation.py

Description: prediction and evaluation

Author: junghwan lee
Email: jhrrlee@gmail.com
Date Created: 2023.10.14

"""


# In[11]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf

def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = 1e-10  # some small constant
    return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + epsilon))) * 100
  
def evaluate_predictions(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred).numpy()  # Convert tensor to numpy value

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape
    }

# cycle by cycle RUL prediction
def RUL_predict(model, x, x_scaler, y_scaler):
  x_norm = x_scaler.transform(x)
  y_pred_norm = model.predict(x_norm)

  y_pred_float = y_scaler.inverse_transform(y_pred_norm)
  y_pred = np.round(y_pred_float).astype(int)

  return y_pred


# In[ ]:


# x (no. of cells * no. of cycles, no. of features), true_ruls (no. of cells * no. of cycles)
"""
class PredictEvaluation:
    __init__(self, trained_model, x, true_ruls, no_of_cells, no_of_cycles, x_scaler, y_scaler):
    self.visual_evaluation = VisualEvaluation(true_ruls,
                                              RUL_predict(trained_model, x, x_scaler, y_scaler),
                                              no_of_cells,
                                              no_of_cycles)

    def metrics(self):
      self.visual_evaluation.metrics()
    def statistic(self):
      self.visual_evaluation.statistic()

"""


# In[1]:


## Visualization class, just existing functions
class VisualEvaluation:
  def __init__(self, y_true, y_pred, no_of_cells, no_of_cycles):
    self.y_true = y_true
    self.y_pred = y_pred
    self.no_of_cells = no_of_cells
    self.no_of_cycles = no_of_cycles
    
    # Reshape the predictions and true values
    y_true_reshaped = y_true.reshape(no_of_cells, no_of_cycles)
    y_pred_reshaped = y_pred.reshape(no_of_cells, no_of_cycles)

    # Initialize lists to store metrics for each cell
    self.mses, self.rmses, self.maes, self.mapes = [], [], [], []

    # Compute metrics for each cell
    for i in range(no_of_cells):
        y_true_cell = y_true_reshaped[i]
        y_pred_cell = y_pred_reshaped[i]

        mse = mean_squared_error(y_true_cell, y_pred_cell)
        rmse = mean_squared_error(y_true_cell, y_pred_cell, squared=False)
        mae = mean_absolute_error(y_true_cell, y_pred_cell)
        mape = mean_absolute_percentage_error(y_true_cell, y_pred_cell).numpy()  # Convert tensor to numpy value

        self.mses.append(mse)
        self.rmses.append(rmse)
        self.maes.append(mae)
        self.mapes.append(mape)
    
    # Compute errors (residuals) for each cell
    residuals = y_true_reshaped - y_pred_reshaped

    # Initialize lists to store metrics for each cell
    self.means, self.medians, self.std_devs, self.iqrs = [], [], [], []

    # Compute metrics for each cell
    for i in range(no_of_cells):
        residuals_cell = residuals[i]

        mean = np.abs(np.mean(residuals_cell))
        median = np.abs(np.median(residuals_cell))
        std_dev = np.std(residuals_cell)
        iqr = np.percentile(residuals_cell, 75) - np.percentile(residuals_cell, 25)

        self.means.append(mean)
        self.medians.append(median)
        self.std_devs.append(std_dev)
        self.iqrs.append(iqr)
  
  def detect_anomalies(self, metric_type, threshold):
    if metric_type not in ['mses', 'rmses', 'maes', 'mapes', 'means', 'medians', 'std_devs', 'iqrs']:
        raise ValueError("Invalid metric type. Choose one of ['mses', 'rmses', 'maes', 'mapes', 'means', 'medians', 'std_devs', 'iqrs'].")

    metric_values = getattr(self, metric_type)
    anomalies = [i for i, value in enumerate(metric_values) if value > threshold]

    if anomalies:
        print(f"Anomalies detected for {metric_type} at cell indices:", anomalies)
    else:
        print(f"No anomalies detected for {metric_type} with threshold {threshold}.")

    return anomalies

  def metrics(self):
    mses, rmses, maes, mapes = metrics_per_cell(self.no_of_cells, 
                                                self.mses,
                                                self.rmses,
                                                self.maes,
                                                self.mapes)
    return mses, rmses, maes, mapes

  def statistic(self):
    means, medians, std_devs, iqrs = statistical_analysis_per_cell(self.no_of_cells,
                                                                   self.means,
                                                                   self.medians,
                                                                   self.std_devs,
                                                                   self.iqrs)
    return means, medians, std_devs, iqrs
  def heatmap(self,
              title = 'Predicted RULs by ResNet',
              cmap_color = 'viridis',
              title_fontsize=16,
              label_fontsize=14, 
              tick_fontsize=12, 
              cbar_fontsize=12, dpi=300, max_xticks=6, max_yticks=6):
    ruls_heatmap(self.y_true,
                      self.y_pred,
                      self.no_of_cells,
                      self.no_of_cycles,                       
                      title = title,
                      cmap_color = cmap_color,
                      title_fontsize=title_fontsize, label_fontsize=label_fontsize, tick_fontsize=tick_fontsize, 
                      cbar_fontsize=cbar_fontsize, dpi=dpi, max_xticks=max_xticks, max_yticks=max_yticks)
    
  def heatmap_separate(self, 
              title = 'Predicted RULs by ResNet',
              title_fontsize=16,
              label_fontsize=14, 
              tick_fontsize=12, 
              cbar_fontsize=12, dpi=300, max_xticks=6, max_yticks=6):
    ruls_heatmap_separate(self.y_true,
                      self.y_pred,
                      self.no_of_cells,
                      self.no_of_cycles, 
                      title = title,
                      title_fontsize=title_fontsize, label_fontsize=label_fontsize, tick_fontsize=tick_fontsize, 
                      cbar_fontsize=cbar_fontsize, dpi=dpi, max_xticks=max_xticks, max_yticks=max_yticks)
    
  def box(self, 
          title = 'RUL Prediction Errors for Each Cell',
          width = 30,
          height = 6):
    boxplot_errors(self.y_true,
                   self.y_pred,
                   self.no_of_cells,
                   self.no_of_cycles,
                   title = title,
                   width = width,
                   height = height)
    
  def swarm(self):
    swarm_plot(self.y_true,
               self.y_pred,
               self.no_of_cells,
               self.no_of_cycles)



import numpy as np
import matplotlib.pyplot as plt

def metrics_per_cell(no_of_cells, mses, rmses, maes, mapes):
    # Visualization
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.bar(range(no_of_cells), mses)
    plt.title('MSE per Cell')

    plt.subplot(2, 2, 2)
    plt.bar(range(no_of_cells), rmses)
    plt.title('RMSE per Cell')

    plt.subplot(2, 2, 3)
    plt.bar(range(no_of_cells), maes)
    plt.title('MAE per Cell')

    plt.subplot(2, 2, 4)
    plt.bar(range(no_of_cells), mapes)
    plt.title('MAPE per Cell')

    plt.tight_layout()
    plt.show()

    return mses, rmses, maes, mapes


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def statistical_analysis_per_cell(no_of_cells, means, medians, std_devs, iqrs):

    # Visualization
    plt.figure(figsize=(20, 15))

    plt.subplot(2, 2, 1)
    plt.bar(range(1, no_of_cells+1), means)
    plt.title('Mean Error per Cell')

    plt.subplot(2, 2, 2)
    plt.bar(range(1, no_of_cells+1), medians)
    plt.title('Median Error per Cell')

    plt.subplot(2, 2, 3)
    plt.bar(range(1, no_of_cells+1), std_devs)
    plt.title('Standard Deviation of Error per Cell')

    plt.subplot(2, 2, 4)
    plt.bar(range(1, no_of_cells+1), iqrs)
    plt.title('IQR per Cell')

    plt.tight_layout()
    plt.show()

    return means, medians, std_devs, iqrs


# In[ ]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# def ruls_heatmap(y_true, y_pred, no_of_cells, no_of_cycles):
#     # Reshape the flat arrays to 2D arrays
#     y_true_2d = y_true.reshape(no_of_cells, no_of_cycles)
#     y_pred_2d = y_pred.reshape(no_of_cells, no_of_cycles)

#     fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

#     sns.heatmap(y_true_2d, ax=ax1, cmap="YlGnBu", cbar_kws={'label': 'True RUL'})
#     ax1.set_title('True RULs')
#     ax1.set_xlabel('Cycle')
#     ax1.set_ylabel('Cell')

#     sns.heatmap(y_pred_2d, ax=ax2, cmap="YlGnBu", cbar_kws={'label': 'Predicted RUL'})
#     ax2.set_title('Predicted RULs')
#     ax2.set_xlabel('Cycle')
#     ax2.set_ylabel('Cell')

#     plt.tight_layout()
#     plt.show()
    
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# def ruls_heatmap(y_true, y_pred, no_of_cells, no_of_cycles,
#                  title_fontsize=16,
#                  label_fontsize=14,
#                  tick_fontsize=12,
#                  cbar_fontsize=12,
#                  dpi=100,
#                  max_xticks=6,
#                  max_yticks=6):
#     """
#     Plots heatmaps for true and predicted RUL values, forcing numeric
#     x/y tick labels, limiting the number of ticks to avoid overlap,
#     and rotating the x-axis labels.
#     """
#     # Reshape the flat arrays into 2D arrays
#     y_true_2d = y_true.reshape(no_of_cells, no_of_cycles)
#     y_pred_2d = y_pred.reshape(no_of_cells, no_of_cycles)

#     # Create figure
#     fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
#                                    figsize=(12, 5),
#                                    dpi=dpi)

#     # --- First heatmap (True RUL) ---
#     hm1 = sns.heatmap(
#         y_true_2d,
#         ax=ax1,
#         cmap="YlGnBu",
#         cbar_kws={'label': 'True RUL'},
#         xticklabels=range(no_of_cycles),   # Force numeric x-labels
#         yticklabels=range(no_of_cells)     # Force numeric y-labels
#     )
#     ax1.set_title('True RULs', fontsize=title_fontsize, color='black')
#     ax1.set_xlabel('Cycle', fontsize=label_fontsize, color='black')
#     ax1.set_ylabel('Cell', fontsize=label_fontsize, color='black')
#     ax1.tick_params(axis='both', labelsize=tick_fontsize, colors='black')

#     # Limit number of ticks
#     ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=max_xticks))
#     ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=max_yticks))

#     # Rotate x-axis tick labels
#     plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', color='black')

#     cbar1 = hm1.collections[0].colorbar
#     cbar1.ax.tick_params(labelsize=cbar_fontsize, colors='black')

#     # --- Second heatmap (Predicted RUL) ---
#     hm2 = sns.heatmap(
#         y_pred_2d,
#         ax=ax2,
#         cmap="YlGnBu",
#         cbar_kws={'label': 'Predicted RUL'},
#         xticklabels=range(no_of_cycles),
#         yticklabels=range(no_of_cells)
#     )
#     ax2.set_title('Predicted RULs', fontsize=title_fontsize, color='black')
#     ax2.set_xlabel('Cycle', fontsize=label_fontsize, color='black')
#     ax2.set_ylabel('Cell', fontsize=label_fontsize, color='black')
#     ax2.tick_params(axis='both', labelsize=tick_fontsize, colors='black')

#     ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=max_xticks))
#     ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=max_yticks))
#     plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', color='black')

#     cbar2 = hm2.collections[0].colorbar
#     cbar2.ax.tick_params(labelsize=cbar_fontsize, colors='black')

#     plt.tight_layout()
#     plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

def ruls_heatmap(y_true, y_pred, no_of_cells, no_of_cycles,
                 title = 'Predicted RULs by ResNet',
                 cmap_color = 'viridis',
                 title_fontsize=16,
                 label_fontsize=14,
                 tick_fontsize=12,
                 cbar_fontsize=12,
                 dpi=300,
                 max_xticks=6,
                 max_yticks=6):
    """
    Plots heatmaps for true and predicted RUL values with custom:
      - title font size
      - x-label and y-label font size
      - x-tick and y-tick font size
      - colorbar font size
      - dpi for higher resolution
    """
    # Reshape the flat arrays to 2D arrays
    y_true_2d = y_true.reshape(no_of_cells, no_of_cycles)
    y_pred_2d = y_pred.reshape(no_of_cells, no_of_cycles)

    # Create a figure with high dpi
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                   figsize=(15, 6),
                                   dpi=dpi)

    # --- First Heatmap (True RUL) ---    
    hm1 = sns.heatmap(
        y_true_2d,
        ax=ax1,
        cmap=cmap_color,
        cbar_kws={'label': 'True RUL'},
        xticklabels=max_xticks,  # Shows every 5th x-label
        yticklabels=max_yticks,   # Shows every 5th y-label
    )
    
    ax1.set_title('True RULs', fontsize=title_fontsize)
    ax1.set_xlabel('Cycle Index', fontsize=label_fontsize)
    ax1.set_ylabel('Cell Index', fontsize=label_fontsize)
    ax1.tick_params(axis='y', labelrotation=90, labelsize=tick_fontsize)
    ax1.tick_params(axis='x', labelsize=tick_fontsize)
    

    cbar1 = hm1.collections[0].colorbar
    cbar1.ax.tick_params(labelsize=cbar_fontsize)
    cbar1.set_label('True RUL', fontsize=cbar_fontsize)
    # ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

    # # Limit y-axis to 6 ticks
    # ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

    # --- Second Heatmap (Predicted RUL) ---
    hm2 = sns.heatmap(
        y_pred_2d,
        ax=ax2,
        cmap=cmap_color,
        cbar_kws={'label': 'Predicted RUL'},
        xticklabels=max_xticks,  # Shows every 5th x-label
        yticklabels=max_yticks   # Shows every 5th y-label
    )
    ax2.set_title(title, fontsize=title_fontsize)
    ax2.set_xlabel('Cycle Index', fontsize=label_fontsize)
    ax2.set_ylabel('Cell Index', fontsize=label_fontsize)
    ax2.tick_params(axis='both', labelsize=tick_fontsize)
    ax2.tick_params(axis='y', labelrotation=90, labelsize=tick_fontsize)
    ax2.tick_params(axis='x', labelsize=tick_fontsize)

    cbar2 = hm2.collections[0].colorbar
    cbar2.ax.tick_params(labelsize=cbar_fontsize)
    cbar2.set_label('Predicted RUL', fontsize=cbar_fontsize)
    # plt.yticks(rotation=90)
    # fig.canvas.draw()
    # plt.setp(ax1.get_yticklabels(), rotation=90, ha='center')
    # plt.setp(ax2.get_yticklabels(), rotation=90, ha='center')
    plt.tight_layout()
    plt.show()

def ruls_heatmap_separate(y_true, y_pred, no_of_cells, no_of_cycles,
                          title = 'Predicted RULs by ResNet',
                          title_fontsize=16,
                          label_fontsize=14,
                          tick_fontsize=12,
                          cbar_fontsize=12,
                          dpi=300,
                          max_xticks=6,
                          max_yticks=6):
    """
    Plots two separate heatmaps (True RUL and Predicted RUL) with custom settings:
      - title font size
      - x-label and y-label font size
      - x-tick and y-tick font size
      - colorbar font size
      - dpi for higher resolution
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Reshape the flat arrays to 2D arrays
    y_true_2d = y_true.reshape(no_of_cells, no_of_cycles)
    y_pred_2d = y_pred.reshape(no_of_cells, no_of_cycles)

    # --- Figure 1: True RUL Heatmap ---
    fig1, ax1 = plt.subplots(figsize=(15, 6), dpi=dpi)
    hm1 = sns.heatmap(
        y_true_2d,
        ax=ax1,
        cmap="YlGnBu",
        cbar_kws={'label': 'True RUL'},
        xticklabels=max_xticks,
        yticklabels=max_yticks,
    )
    ax1.set_title('True RULs', fontsize=title_fontsize)
    ax1.set_xlabel('Cycle Index', fontsize=label_fontsize)
    ax1.set_ylabel('Cell Index', fontsize=label_fontsize)
    ax1.tick_params(axis='x', labelsize=tick_fontsize)
    ax1.tick_params(axis='y', labelrotation=90, labelsize=tick_fontsize)

    # Adjust the colorbar
    cbar1 = hm1.collections[0].colorbar
    cbar1.ax.tick_params(labelsize=cbar_fontsize)
    cbar1.set_label('True RUL', fontsize=cbar_fontsize)
    
    plt.tight_layout()
    plt.show()

    # --- Figure 2: Predicted RUL Heatmap ---
    fig2, ax2 = plt.subplots(figsize=(15, 6), dpi=dpi)
    hm2 = sns.heatmap(
        y_pred_2d,
        ax=ax2,
        cmap="YlGnBu",
        cbar_kws={'label': 'Predicted RUL'},
        xticklabels=max_xticks,
        yticklabels=max_yticks
    )
    ax2.set_title('Predicted RULs', fontsize=title_fontsize)
    ax2.set_xlabel('Cycle Index', fontsize=label_fontsize)
    ax2.set_ylabel('Cell Index', fontsize=label_fontsize)
    ax2.tick_params(axis='x', labelsize=tick_fontsize)
    ax2.tick_params(axis='y', labelrotation=90, labelsize=tick_fontsize)

    # Adjust the colorbar
    cbar2 = hm2.collections[0].colorbar
    cbar2.ax.tick_params(labelsize=cbar_fontsize)
    cbar2.set_label('Predicted RUL', fontsize=cbar_fontsize)
    
    plt.tight_layout()
    plt.show()


import seaborn as sns
import numpy as np

# def boxplot_errors(y_true, y_pred, no_of_cells, no_of_cycles):
#     # Assuming y_true and y_pred are 1D arrays with shape (no_of_cells * no_of_cycles,)
#     y_true = y_true.reshape(no_of_cells, no_of_cycles)
#     y_pred = y_pred.reshape(no_of_cells, no_of_cycles)

#     # Compute errors for each cell across all cycles
#     errors = y_true - y_pred

#     # Flatten errors and create a cell label for each error
#     flat_errors = errors.flatten()
#     cell_labels = np.repeat(np.arange(no_of_cells), no_of_cycles)

#     plt.figure(figsize=(30, 6))

#     # Box plot
#     #sns.boxplot(x=cell_labels, y=flat_errors, color='skyblue', showfliers=False)  # showfliers=False removes outliers for clarity
#     #sns.boxplot(data=residuals)
#     ax = sns.boxplot(x=cell_labels, y=flat_errors)
#     ticks = list(range(0, no_of_cells, 10))  # Show every 10th label
#     ax.set_xticks(ticks)
#     ax.set_xticklabels(ticks, fontsize=10)  # You can adjust the fontsize value as needed
#     # Swarm plot
#     #sns.swarmplot(x=cell_labels, y=flat_errors, color='red', size=2.5)

#     plt.xlabel('Cell Number')
#     plt.ylabel('Errors (cycle)')
#     plt.title(f'{no_of_cycles} Cycle Distribution of Cycle by Cycle RUL Prediction Errors for Each Cell')
#     plt.show()

def boxplot_errors(y_true, y_pred, no_of_cells, no_of_cycles, 
                   title = 'RUL Prediction Errors for Each Cell',
                   width = 30,
                   height = 6):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Reshape data
    y_true = y_true.reshape(no_of_cells, no_of_cycles)
    y_pred = y_pred.reshape(no_of_cells, no_of_cycles)
    errors = y_true - y_pred

    # Flatten errors and create a cell label for each error
    flat_errors = errors.flatten()
    cell_labels = np.repeat(np.arange(no_of_cells), no_of_cycles)

    # Set a pleasing style and context for larger fonts
    sns.set_style("whitegrid")
    sns.set_context("talk")  # scales fonts and elements for presentations

    # Set figure size
    plt.figure(figsize=(width, height))

    # Create the box plot with a custom color palette for enhanced visual appeal
    ax = sns.boxplot(x=cell_labels, y=flat_errors, palette="pastel")

    # Customize x-ticks: show every 10th label and adjust font size
    ticks = list(range(0, no_of_cells, 10))
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, fontsize=20)
    ax.tick_params(axis='y', labelsize=20)  # adjust y tick label size

    # Customize axis labels and title with font sizes
    plt.xlabel('Cell Index', fontsize=20)
    plt.ylabel('Errors (cycle)', fontsize=20)
    plt.title(title, fontsize=20)

    plt.show()


import seaborn as sns
import numpy as np

def swarm_plot(y_true, y_pred, no_of_cells, no_of_cycles):
    # Assuming y_true and y_pred are 1D arrays with shape (no_of_cells * no_of_cycles,)
    y_true = y_true.reshape(no_of_cells, no_of_cycles)
    y_pred = y_pred.reshape(no_of_cells, no_of_cycles)

    # Compute errors for each cell across all cycles
    errors = y_true - y_pred

    # Flatten errors and create a cell label for each error
    flat_errors = errors.flatten()
    cell_labels = np.repeat(np.arange(no_of_cells), no_of_cycles)

    plt.figure(figsize=(30, 6))

    # Box plot
    #sns.boxplot(x=cell_labels, y=flat_errors, color='skyblue', showfliers=False)  # showfliers=False removes outliers for clarity
    #sns.boxplot(data=residuals)
    #sns.boxplot(x=cell_labels, y=flat_errors)
    # Swarm plot
    ax = sns.swarmplot(x=cell_labels, y=flat_errors, color='red', size=1)
    ticks = list(range(0, no_of_cells, 10))  # Show every 10th label
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, fontsize=10)  # You can adjust the fontsize value as needed

    plt.xlabel('Cell Number')
    plt.ylabel('Errors (cycle)')
    plt.title(f'{no_of_cycles} Cycle Distribution of Cycle by Cycle RUL Prediction Errors for Each Cell')
    plt.show()

