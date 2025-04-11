import importlib.util

# Ensuring required libraries are available
for lib in ['numpy', 'pandas']:
    if importlib.util.find_spec(lib) is None:
        raise ImportError(f"The required library '{lib}' is not installed. Please install it using 'pip install {lib}'.")


import numpy as np
import pandas as pd


class GeometricRegressor:
    def __init__(self):
        self.weights = {}
        self.angles = {}
        self.dist_ratio = {}
        self.area_perimeter_ratio = {}
        self.params_ = {}  # To hold all parameters after fitting

    def fit(self, X, y):
        """Fit the model by computing geometric parameters."""
        self.X_train = X
        self.y_train = y

        # Convert y to a pandas Series if it's not already.
        if not isinstance(y, pd.Series):
          y = pd.Series(y, index=X.index)  # Assume y has the same index as X
        # Compute weights based on correlation for each column.
        for col in X.columns:
            corr_ = X[col].corr(y)
            self.weights[col] = abs(corr_)
        
        # Normalize weights so they sum to 1.
        weight_sum = sum(self.weights.values())
        if weight_sum > 0:
            for col in self.weights:
                self.weights[col] /= weight_sum

        # Compute geometric parameters.
        self.angles = self._compute_angles(X, y)
        self.dist_ratio = self._compute_distance_ratios(X, y)
        self.area_perimeter_ratio = self._compute_area_perimeter_ratios(X, y)
        
        # Bundle all parameters into one attribute.
        self.params_ = {
            'weights': self.weights,
            'angles': self.angles,
            'dist_ratio': self.dist_ratio,
            'area_perimeter_ratio': self.area_perimeter_ratio
        }

    def predict(self, X_val, return_intermediates=False, detail_column=None, detail_range=None):
        """
        Predict using the geometric parameters and the column-first approach.
        
        Parameters:
            X_val: pandas DataFrame with validation data.
            return_intermediates: bool (default False).
                If True, returns a tuple (final_predictions, preds_df, detailed_preds)
                where:
                  - final_predictions: pandas Series of final predictions.
                  - preds_df: DataFrame with per-column predictions for all rows.
                  - detailed_preds: DataFrame with intermediate predictions for a few datapoints 
                                    per selected column.
            detail_column: list of str, optional.
                List of column names for which detailed predictions are returned.
                If None, detailed data for one datapoint from each column is returned.
            detail_range: int or sequence, optional.
                If an integer n is provided, the top n datapoints are used.
                If a sequence (e.g., list of row indices) is provided, only those rows are used.
        
        Returns:
            If return_intermediates is False, returns a pandas Series of final predictions.
            Otherwise, returns a tuple: (final_predictions, preds_df, detailed_preds)
        """
        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
            raise ValueError("Model has not been fitted yet.")
        
        # Compute per-column predictions for each row.
        preds_df = pd.DataFrame(index=X_val.index)
        for col in X_val.columns:
            Q_val = X_val[col].values.astype(float)
            angle = self.angles.get(col, 0)
            pred_angle = Q_val * np.tan(angle)
            
            ratio = self.dist_ratio.get(col, 1)
            pred_dist = np.where(Q_val == 0, 0, np.sqrt((ratio * np.abs(Q_val))**2 - Q_val**2))
            
            apr = self.area_perimeter_ratio.get(col, 0)
            pred_apr = np.vectorize(self._predict_from_apr)(Q_val, apr)
            
            col_pred = (pred_angle + pred_dist + pred_apr) / 3.0
            preds_df[col] = col_pred
        
        # Final prediction: weighted average across columns.
        final_pred = np.zeros(len(X_val))
        for col in X_val.columns:
            weight = self.weights.get(col, 0)
            final_pred += weight * preds_df[col].values
        final_predictions = pd.Series(final_pred, index=X_val.index)
        
        if not return_intermediates:
            return final_predictions
        else:
            # Determine which columns to include in detailed predictions.
            all_cols = list(X_val.columns)
            if detail_column is not None:
                # Validate columns.
                for col in detail_column:
                    if col not in all_cols:
                        raise ValueError(f"Column '{col}' is not present in the validation data.")
                selected_cols = detail_column
            else:
                # By default, return one datapoint per column.
                selected_cols = all_cols
            
            # For each selected column, pick the datapoints based on detail_range.
            # If detail_range is not provided, use the first datapoint.
            detailed_frames = []
            for col in selected_cols:
                Q_vals = X_val[col].values.astype(float)
                angle = self.angles.get(col, 0)
                pred_angle = Q_vals * np.tan(angle)
                
                ratio = self.dist_ratio.get(col, 1)
                pred_dist = np.where(Q_vals == 0, 0, np.sqrt((ratio * np.abs(Q_vals))**2 - Q_vals**2))
                
                apr = self.area_perimeter_ratio.get(col, 0)
                pred_apr = np.vectorize(self._predict_from_apr)(Q_vals, apr)
                
                # Column-level prediction (before weighted aggregation)
                col_pred = (pred_angle + pred_dist + pred_apr) / 3.0
                
                # Now, select the rows for detailed output.
                if detail_range is None:
                    # Use only the first datapoint.
                    indices = [X_val.index[0]]
                elif isinstance(detail_range, int):
                    indices = X_val.index[:detail_range]
                else:
                    # Assume detail_range is a sequence of indices.
                    indices = detail_range
                
                # Create a DataFrame for this column.
                df = pd.DataFrame({
                    'X_val_datapoint': Q_vals,
                    'angle_pred': pred_angle,
                    'dist_ratio_pred': pred_dist,
                    'apr_pred': pred_apr,
                    'col_pred': col_pred  # per-column prediction
                }, index=X_val.index)
                df = df.loc[indices].copy()
                # Compute the final predicted value for each row using the weighted average
                # across all columns (from final_predictions).
                df['predicted_value'] = final_predictions.loc[indices]
                df['feature'] = col
                # Reset index so that the feature is used as the index.
                df = df.reset_index(drop=True)
                df.index = [col] * len(df)
                detailed_frames.append(df)
            
            detailed_preds = pd.concat(detailed_frames)
            return final_predictions, preds_df, detailed_preds

    def _predict_from_apr(self, Q, apr):
        """
        Predict the target value W for a given Q and area-perimeter ratio (apr)
        by solving the equation:
            apr = (0.5 * |Q| * W) / (|Q| + W + sqrt(Q^2 + W^2))
        Uses an iterative gradient-descent-like method.
        """
        if Q == 0:
            return 0.0
        W = np.abs(Q)
        lr = 0.01
        for _ in range(50):
            perimeter = np.abs(Q) + W + np.sqrt(Q**2 + W**2)
            current_apr = (0.5 * np.abs(Q) * W) / perimeter if perimeter > 0 else 0
            error = current_apr - apr
            delta = 1e-6
            perimeter_delta = np.abs(Q) + (W + delta) + np.sqrt(Q**2 + (W + delta)**2)
            apr_plus = (0.5 * np.abs(Q) * (W + delta)) / perimeter_delta if perimeter_delta > 0 else 0
            derivative = (apr_plus - current_apr) / delta
            if derivative == 0:
                break
            W -= ((lr * error) / derivative)
        return W

    def _compute_angles(self, X_to_fit, y_to_fit):
        """Compute the mean angle for each column in X_to_fit relative to y_to_fit."""
        angles_dict = {}
        for col in X_to_fit.columns:
            Q = X_to_fit[col]
            W = y_to_fit
            angles = np.where(Q == 0, np.pi/2, np.arctan(W / Q))
            angles_dict[col] = np.mean(angles)
        return angles_dict

    def _compute_distance_ratios(self, X_to_fit, y_to_fit):
        """Compute the mean distance ratio for each column in X_to_fit relative to y_to_fit."""
        ratio_dict = {}
        for col in X_to_fit.columns:
            Q = X_to_fit[col]
            W = y_to_fit
            distance = np.sqrt(Q**2 + W**2)
            ratio = np.where(Q == 0, np.where(W != 0, np.inf, 0), distance / np.abs(Q))
            ratio_dict[col] = np.mean(ratio)
        return ratio_dict

    def _compute_area_perimeter_ratios(self, X_to_fit, y_to_fit):
        """Compute the mean area-to-perimeter ratio for each column in X_to_fit relative to y_to_fit."""
        ratio_dict = {}
        for col in X_to_fit.columns:
            Q = X_to_fit[col]
            W = y_to_fit
            area = 0.5 * np.abs(Q) * np.abs(W)
            perimeter = np.abs(Q) + np.abs(W) + np.sqrt(Q**2 + W**2)
            ratio = np.where(perimeter > 0, area / perimeter, 0)
            ratio_dict[col] = np.mean(ratio)
        return ratio_dict
