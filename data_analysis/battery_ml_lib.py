# -*- coding: utf-8 -*-
"""
File Name: battery_ml_lib.py

Description: analysis of features.

Author: junghwan lee
Email: jhrrlee@gmail.com
Date Created: 2023.09.08
"""

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
import xgboost as xgb
#import shap
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# class BatteryFeatureImportance:
#     def __init__(self, new_cycle_sum, eol, feature_names):
#         self.new_cycle_sum = new_cycle_sum
#         self.eol = eol.ravel()
#         self.feature_names = feature_names

#         self.num_cells, self.num_features, self.num_cycles = new_cycle_sum.shape
#         self.expanded_data = new_cycle_sum.reshape(self.num_cells, -1)

#         self.model = None
#         self.X_train = None
#         self.X_test = None
#         self.Y_train = None
#         self.Y_test = None

#         # For storing permutation importance and aggregated importance
#         self.perm_importance = None
#         self.aggregated_importance = None

#     def train_model(self, regressor_choice='random_forest', n_estimators=100):
#         self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.expanded_data, self.eol, test_size=0.3, random_state=42)

#         if regressor_choice == 'random_forest':
#             self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
#         elif regressor_choice == 'gradient_boosting':
#             self.model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
#         elif regressor_choice == 'lasso':
#             self.model = Lasso(alpha=0.1)
#         elif regressor_choice == 'xgboost':
#             self.model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=42)
#         else:
#             raise ValueError(f"Unknown regressor choice: {regressor_choice}")

#         self.model.fit(self.X_train, self.Y_train)

#     def feature_importance(self, method='permutation', n_repeats=30):
#         if method == 'permutation':
#             perm_result = permutation_importance(self.model, self.X_test, self.Y_test, n_repeats=n_repeats, random_state=42)
#             self.perm_importance = perm_result.importances_mean
#         elif method == 'coefficients':
#             self.perm_importance = self.model.coef_
#         elif method == 'tree_importance':
#             self.perm_importance = self.model.feature_importances_
#         elif method == 'shap':
#             explainer = shap.TreeExplainer(self.model)
#             shap_values = explainer.shap_values(self.X_test)
#             self.perm_importance = np.mean(np.abs(shap_values), axis=0)
#         else:
#             raise ValueError(f"Unknown feature importance method: {method}")

#         self.aggregate_feature_importance()
#         self.plot_aggregated_importance()

#     def evaluate_model(self):
#         train_preds = self.model.predict(self.X_train)
#         test_preds = self.model.predict(self.X_test)

#         metrics = {
#             "Training MSE": mean_squared_error(self.Y_train, train_preds),
#             "Validation MSE": mean_squared_error(self.Y_test, test_preds),
#             "Training RMSE": np.sqrt(mean_squared_error(self.Y_train, train_preds)),
#             "Validation RMSE": np.sqrt(mean_squared_error(self.Y_test, test_preds)),
#             "Training MAE": mean_absolute_error(self.Y_train, train_preds),
#             "Validation MAE": mean_absolute_error(self.Y_test, test_preds),
#             "Training MAPE": np.mean(np.abs((self.Y_train - train_preds) / self.Y_train)) * 100,
#             "Validation MAPE": np.mean(np.abs((self.Y_test - test_preds) / self.Y_test)) * 100
#         }

#         for metric, value in metrics.items():
#             print(f"{metric}: {value:.4f}")

#     def aggregate_feature_importance(self):
#         expanded_importance = self.perm_importance
#         aggregated_importance = [expanded_importance[i::self.num_cycles].sum() for i in range(self.num_features)]
#         self.aggregated_importance = np.array(aggregated_importance)

# #     def plot_aggregated_importance(self):
#         sorted_idx = self.aggregated_importance.argsort()
#         feature_names_sorted = np.array(self.feature_names)[sorted_idx]
#         feature_importances_sorted = self.aggregated_importance[sorted_idx]

#         plt.figure(figsize=(10, 5))
#         plt.barh(feature_names_sorted, feature_importances_sorted)
#         plt.xlabel('Importance')
#         plt.ylabel('Features')
#         plt.title('Feature Importance')
#         plt.show()
                
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb

class BatteryFeatureImportance:
    def __init__(self, new_cycle_sum, eol, feature_names):
        """
        :param new_cycle_sum: np.array of shape (num_cells, num_features, num_cycles)
        :param eol: np.array of shape (num_cells,) or (num_cells, 1) - end of life values
        :param feature_names: list of length num_features with feature names
        """
        self.new_cycle_sum = new_cycle_sum
        self.eol = eol.ravel()
        self.feature_names = feature_names

        self.num_cells, self.num_features, self.num_cycles = new_cycle_sum.shape
        # Flatten (num_cells, num_features, num_cycles) to (num_cells, num_features * num_cycles)
        self.expanded_data = new_cycle_sum.reshape(self.num_cells, -1)

        self.model = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

        # For storing permutation importance and aggregated importance
        self.perm_importance = None
        self.aggregated_importance = None

    def train_model(self, regressor_choice='random_forest', n_estimators=100):
        """
        Splits data into train/test and fits a specified regression model.
        """
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.expanded_data, self.eol, test_size=0.3, random_state=42
        )

        if regressor_choice == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        elif regressor_choice == 'gradient_boosting':
            self.model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
        elif regressor_choice == 'lasso':
            self.model = Lasso(alpha=0.1)
        elif regressor_choice == 'xgboost':
            self.model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=42)
        else:
            raise ValueError(f"Unknown regressor choice: {regressor_choice}")

        self.model.fit(self.X_train, self.Y_train)

    def feature_importance(self, method='permutation', n_repeats=30):
        """
        Computes feature importance using one of the following methods:
          - 'permutation': scikit-learn permutation_importance
          - 'coefficients': model coefficients (for linear models)
          - 'tree_importance': feature_importances_ from tree-based models
          - 'shap': mean(|SHAP values|)
        Then aggregates and plots the results.
        """
        if not self.model:
            raise RuntimeError("Model must be trained before computing feature importance.")

        if method == 'permutation':
            perm_result = permutation_importance(
                self.model, self.X_test, self.Y_test, 
                n_repeats=n_repeats, random_state=42
            )
            self.perm_importance = perm_result.importances_mean
        elif method == 'coefficients':
            # For linear models like Lasso
            self.perm_importance = self.model.coef_
        elif method == 'tree_importance':
            # For tree-based models like RandomForest, XGBoost
            self.perm_importance = self.model.feature_importances_
        elif method == 'shap':
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.X_test)
            self.perm_importance = np.mean(np.abs(shap_values), axis=0)
        else:
            raise ValueError(f"Unknown feature importance method: {method}")

        self.aggregate_feature_importance()
        self.plot_aggregated_importance()

    def evaluate_model(self):
        """
        Prints out several error metrics for training and validation sets.
        """
        if not self.model:
            raise RuntimeError("Model must be trained before evaluation.")

        train_preds = self.model.predict(self.X_train)
        test_preds = self.model.predict(self.X_test)

        metrics = {
            "Training MSE": mean_squared_error(self.Y_train, train_preds),
            "Validation MSE": mean_squared_error(self.Y_test, test_preds),
            "Training RMSE": np.sqrt(mean_squared_error(self.Y_train, train_preds)),
            "Validation RMSE": np.sqrt(mean_squared_error(self.Y_test, test_preds)),
            "Training MAE": mean_absolute_error(self.Y_train, train_preds),
            "Validation MAE": mean_absolute_error(self.Y_test, test_preds),
            "Training MAPE": np.mean(np.abs((self.Y_train - train_preds) / self.Y_train)) * 100,
            "Validation MAPE": np.mean(np.abs((self.Y_test - test_preds) / self.Y_test)) * 100
        }

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    def aggregate_feature_importance(self):
        """
        Sums the (permutation or other) importance values across all cycles
        for each feature. This assumes self.perm_importance is a 1D array
        of length (num_features * num_cycles).
        """
        if self.perm_importance is None:
            raise RuntimeError("Permutation/feature importance must be computed before aggregation.")

        aggregated_importance = []
        # Each block of size self.num_cycles corresponds to a single feature
        for f in range(self.num_features):
            start_idx = f * self.num_cycles
            end_idx = (f + 1) * self.num_cycles
            feature_sum = self.perm_importance[start_idx:end_idx].sum()
            aggregated_importance.append(feature_sum)

        self.aggregated_importance = np.array(aggregated_importance)
        

    def plot_aggregated_importance(self, method_name="Permutation"):
        """
        Plots feature importances in the original (fixed) feature order.
        Highlights certain features in red, removes the y-label,
        and places RANK values (e.g. #1, #2, ...) at the end of each bar.
        """

        if self.aggregated_importance is None:
            raise RuntimeError("Must aggregate feature importance before plotting.")

        # 1) Keep features in the original order (no sorting):
        feature_names_ordered = self.feature_names
        feature_importances_ordered = self.aggregated_importance

        # 2) Decide which features to highlight in red:
        highlight_features = {"ICavg", "ICmax", "IDavg", "IDmax"} 

        # Build a list of colors: red if feature is in highlight_features, otherwise blue
        bar_colors = [
            "red" if f in highlight_features else "#1f77b4"
            for f in feature_names_ordered
        ]

        # 3) Compute ranks based on importance (rank=1 for the highest importance)
        #    We'll create an array 'ranks' that is aligned with the original order.
        #    Steps:
        #      - Sort indices by descending importance
        #      - Assign ranks (1-based) in that order
        desc_order = np.argsort(-feature_importances_ordered)  # indices sorted descending
        ranks = np.zeros_like(feature_importances_ordered, dtype=int)
        for rank, idx in enumerate(desc_order, start=1):
            ranks[idx] = rank  # rank 1 = highest importance, 2 = second, etc.

        fig, ax = plt.subplots(figsize=(6, 4))

        # 4) Create a horizontal bar plot in the given (fixed) order
        bars = ax.barh(feature_names_ordered, feature_importances_ordered, color=bar_colors)

        # Remove the y-axis label (the axis line/ticks remain, just no label text)
        ax.set_ylabel("")

        # x-label and title can depend on the method or your preference
        ax.set_xlabel(f"{method_name}-based Feature Score")
        # ax.set_title(f"Feature Importance - {method_name}")

        # 5) Place RANK values at the end of each bar (e.g. "#1", "#2", ...)
        for bar, rank_val in zip(bars, ranks):
            width = bar.get_width()
            ax.text(
                width * 1.01,                     # x-position: slightly to the right of the bar
                bar.get_y() + bar.get_height() / 2,
                f"#{rank_val}",                   # show rank instead of numeric importance
                va='center',
                ha='left',
                fontsize=9
            )

        
        ax.set_xscale("log")
        
        ax.set_xlim(left=max(1e-3, min(feature_importances_ordered) * 0.8))

        plt.tight_layout()
        plt.show()
