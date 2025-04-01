import numpy as np
import pandas as pd

class GeometricRegressor:
    def __init__(self):
        self.weights = {}
        self.angles = {}
        self.dist_ratio = {}
        self.area_perimeter_ratio = {}
        self.params_ = {}  # To hold all the above parameters after fitting

    def fit(self, X, y):
        """Fit the model by computing geometric parameters."""
        self.X_train = X
        self.y_train = y

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

    def predict(self, X_val):
        """Predict using the geometric parameters and the column-first approach."""
        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
            raise ValueError("Model has not been fitted yet.")
        
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
        
        final_pred = np.zeros(len(X_val))
        for col in X_val.columns:
            weight = self.weights.get(col, 0)
            final_pred += weight * preds_df[col].values
        
        return pd.Series(final_pred, index=X_val.index)

    def _predict_from_apr(self, Q, apr):
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

    def _compute_angles(self, X, y):
        angles_dict = {}
        for col in X.columns:
            Q = X[col]
            W = y
            angles = np.where(Q == 0, np.pi/2, np.arctan(W / Q))
            angles_dict[col] = np.mean(angles)
        return angles_dict

    def _compute_distance_ratios(self, X, y):
        ratio_dict = {}
        for col in X.columns:
            Q = X[col]
            W = y
            distance = np.sqrt(Q**2 + W**2)
            ratio = np.where(Q == 0, np.where(W != 0, np.inf, 0), distance / np.abs(Q))
            ratio_dict[col] = np.mean(ratio)
        return ratio_dict

    def _compute_area_perimeter_ratios(self, X, y):
        ratio_dict = {}
        for col in X.columns:
            Q = X[col]
            W = y
            area = 0.5 * np.abs(Q) * np.abs(W)
            perimeter = np.abs(Q) + np.abs(W) + np.sqrt(Q**2 + W**2)
            ratio = np.where(perimeter > 0, area / perimeter, 0)
            ratio_dict[col] = np.mean(ratio)
        return ratio_dict
