import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from scipy.stats import zscore

class GeometricRegressor:
    def __init__(self, detect_outliers=False, outlier_method='zscore', z_threshold=3.0,
                 contamination=0.05, dbscan_eps=0.5, dbscan_min_samples=5, random_state=42):
        self.detect_outliers = detect_outliers
        self.outlier_method = outlier_method
        self.z_threshold = z_threshold
        self.contamination = contamination
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.random_state = random_state

        self.weights = {}
        self.angles = {}
        self.dist_ratio = {}
        self.area_perimeter_ratio = {}
        self.params_ = {}  # To hold all the above parameters after fitting

    def _detect_outliers(self, X):
        """Detects and handles outliers in the dataset based on the chosen method."""
        mask = np.ones(X.shape[0], dtype=bool)  # Initialize all True (all points are inliers)

        if self.outlier_method == 'zscore':
            z_scores = np.abs(zscore(X))
            mask = mask & (z_scores < self.z_threshold).all(axis=1)
        elif self.outlier_method == 'isolation_forest':
            iso_forest = IsolationForest(contamination=self.contamination, random_state=self.random_state)
            mask = (iso_forest.fit_predict(X) == 1)
        elif self.outlier_method == 'dbscan':
            db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
            mask = (db.fit_predict(X) != -1)
        return mask

    def fit(self, X, y):
        """Fit the model, applying outlier detection if opted in."""
        if self.detect_outliers:
            mask = self._detect_outliers(X)
            self.X_train = X[mask]
            self.y_train = y[mask]
        else:
            self.X_train = X
            self.y_train = y

        X_to_fit = self.X_train
        y_to_fit = self.y_train

        # Compute weights based on correlation for each column.
        for col in X_to_fit.columns:
            corr_ = X_to_fit[col].corr(y_to_fit)
            self.weights[col] = abs(corr_)
        
        # Normalize weights so they sum to 1.
        weight_sum = sum(self.weights.values())
        if weight_sum > 0:
            for col in self.weights:
                self.weights[col] /= weight_sum

        # Compute geometric parameters.
        self.angles = self._compute_angles(X_to_fit, y_to_fit)
        self.dist_ratio = self._compute_distance_ratios(X_to_fit, y_to_fit)
        self.area_perimeter_ratio = self._compute_area_perimeter_ratios(X_to_fit, y_to_fit)
        
        # Bundle all parameters into one attribute.
        self.params_ = {
            'weights': self.weights,
            'angles': self.angles,
            'dist_ratio': self.dist_ratio,
            'area_perimeter_ratio': self.area_perimeter_ratio
        }

    def predict(self, X_val):
        """Predict using the geometric parameters and the column-first approach.
        
        For each column in X_val, using the corresponding parameter values (angle, distance ratio,
        and area-perimeter ratio) the function computes three predicted values for the output.
        Their mean is taken as the per-column prediction. Finally, the weighted average across
        columns is computed (using self.weights) to yield the final prediction for each row.
        
        Returns:
            A pandas Series of final predicted values.
        """
        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
            raise ValueError("Model has not been fitted yet.")
        
        # DataFrame to hold per-column predictions.
        preds_df = pd.DataFrame(index=X_val.index)
        
        # Iterate through each column (vectorized over rows)
        for col in X_val.columns:
            Q_val = X_val[col].values.astype(float)
            # Prediction from angle: W = Q * tan(angle)
            angle = self.angles.get(col, 0)
            pred_angle = Q_val * np.tan(angle)
            
            # Prediction from distance ratio: W = sqrt((ratio * |Q|)^2 - Q^2)
            ratio = self.dist_ratio.get(col, 1)
            # When Q is zero, define prediction as 0.
            pred_dist = np.where(Q_val == 0, 0, np.sqrt((ratio * np.abs(Q_val))**2 - Q_val**2))
            
            # Prediction from area-perimeter ratio using a helper solver.
            apr = self.area_perimeter_ratio.get(col, 0)
            # Vectorize the iterative solver.
            pred_apr = np.vectorize(self._predict_from_apr)(Q_val, apr)
            
            # Mean of the three predictions for this column.
            col_pred = (pred_angle + pred_dist + pred_apr) / 3.0
            preds_df[col] = col_pred
        
        # Final prediction: weighted average across columns.
        # Ensure the weights are in the same order as the columns.
        final_pred = np.zeros(len(X_val))
        for col in X_val.columns:
            weight = self.weights.get(col, 0)
            final_pred += weight * preds_df[col].values
        return pd.Series(final_pred, index=X_val.index)

    def _predict_from_apr(self, Q, apr):
        """
        Predicts the target value W for a given Q and area-perimeter ratio (apr).
        
        We need to solve for W in the equation:
          apr = (0.5 * |Q| * W) / (|Q| + W + sqrt(Q^2 + W^2))
        For Q == 0, we define the prediction as 0.
        
        This function uses a simple iterative approach (gradient descent-like) to find W.
        """
        if Q == 0:
            return 0.0
        # Initial guess for W: use |Q|
        W = np.abs(Q)
        lr = 0.01  # learning rate
        for _ in range(50):
            # Compute current area-perimeter ratio using current W.
            perimeter = np.abs(Q) + W + np.sqrt(Q**2 + W**2)
            current_apr = (0.5 * np.abs(Q) * W) / perimeter if perimeter > 0 else 0
            # Compute error.
            error = current_apr - apr
            # A simple derivative approximation (finite differences)
            delta = 1e-6
            perimeter_delta = np.abs(Q) + (W + delta) + np.sqrt(Q**2 + (W + delta)**2)
            apr_plus = (0.5 * np.abs(Q) * (W + delta)) / perimeter_delta if perimeter_delta > 0 else 0
            derivative = (apr_plus - current_apr) / delta
            # Update W; avoid division by zero in derivative.
            if derivative == 0:
                break
            W -= ((lr * error) / derivative)
        return W

    def _compute_angles(self, X_to_fit, y_to_fit):
        """
        Compute the mean angle for each column in X_to_fit relative to y_to_fit.
        For each column, for each row, the angle is computed as arctan(W / Q)
        where Q is the value from the column and W is the corresponding value in y_to_fit.
        If Q is zero, the angle is set to pi/2.
        Returns:
            A dictionary with keys as column names and values as the mean angle (in radians).
        """
        angles_dict = {}
        for col in X_to_fit.columns:
            Q = X_to_fit[col]
            W = y_to_fit
            angles = np.where(Q == 0, np.pi/2, np.arctan(W / Q))
            angles_dict[col] = np.mean(angles)
        return angles_dict

    def _compute_distance_ratios(self, X_to_fit, y_to_fit):
        """
        Compute the mean distance ratio for each column in X_to_fit relative to y_to_fit.
        For each data point, Q is the value from the column and W is the corresponding value in y_to_fit.
        The distance is computed as sqrt(Q^2 + W^2) and the ratio is distance/|Q|.
        For Q == 0, the ratio is set to np.inf if W != 0, else 0.
        Returns:
            A dictionary with keys as column names and values as the mean distance ratio.
        """
        ratio_dict = {}
        for col in X_to_fit.columns:
            Q = X_to_fit[col]
            W = y_to_fit
            distance = np.sqrt(Q**2 + W**2)
            ratio = np.where(Q == 0, np.where(W != 0, np.inf, 0), distance / np.abs(Q))
            ratio_dict[col] = np.mean(ratio)
        return ratio_dict

    def _compute_area_perimeter_ratios(self, X_to_fit, y_to_fit):
        """
        Compute the mean area-to-perimeter ratio for each column in X_to_fit relative to y_to_fit.
        For each data point, Q is the value from the column and W is the corresponding value in y_to_fit.
        The area is calculated as 0.5 * |Q| * |W|, and the perimeter as |Q| + |W| + sqrt(Q^2 + W^2).
        The ratio is then area/perimeter (if perimeter > 0, otherwise 0).
        Returns:
            A dictionary with keys as column names and values as the mean area-to-perimeter ratio.
        """
        ratio_dict = {}
        for col in X_to_fit.columns:
            Q = X_to_fit[col]
            W = y_to_fit
            area = 0.5 * np.abs(Q) * np.abs(W)
            perimeter = np.abs(Q) + np.abs(W) + np.sqrt(Q**2 + W**2)
            ratio = np.where(perimeter > 0, area / perimeter, 0)
            ratio_dict[col] = np.mean(ratio)
        return ratio_dict
