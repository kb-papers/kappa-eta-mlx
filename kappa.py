"""
kappa-eta pipeline — optimized multi-core CPU baseline (Numba JIT)
==================================================================

The baseline the GPU is measured against, Appendix A of the research paper.
Follows the scikit-learn estimator API so it
composes with GridSearchCV and friends.

Only the OUTER loop is parallelised — the encoder across feature columns, the
regressor across test samples. Each core still sweeps all N training points for
its assigned sample (or feature) serially, which is what makes the CPU
effectively saturated for any non-trivial N and gives T_cpu = a_c N^2 P + g_c.
All arithmetic is float32, matching the GPU path so the two are comparable.
"""
import numpy as np
import logging
from numba import njit, prange
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


logger = logging.getLogger(__name__)
logger.propagate = False

@njit(fastmath=True)
def kappa_impute_numba(x_t, x_i, y_i, kappa):
    distances = np.abs(x_t - x_i)
    weights = (1 / ((1 + distances) ** kappa)).astype(np.float32)
    return np.dot(y_i, weights) / np.sum(weights)

@njit(parallel=True)
def _encode_numba_inner(X_encoded, columns, train_col_values, train_target_values, kappa_values):
    n_cols = len(columns)
    n_rows = X_encoded.shape[0]

    for col_idx in prange(n_cols):
        col = columns[col_idx]
        X_train_col_np = train_col_values[col_idx]
        unique_vals_x = np.unique(X_encoded[:, col])
        unique_vals_train = np.unique(X_train_col_np)
        unique_vals = np.unique(np.concatenate((unique_vals_x, unique_vals_train)))

        for val in unique_vals:
            imputed_value = kappa_impute_numba(
                val,
                X_train_col_np,
                train_target_values,
                kappa_values[col_idx]
            )

            for row_idx in range(n_rows):
                if X_encoded[row_idx, col] == val:
                    X_encoded[row_idx, col] = imputed_value

    return X_encoded

class KappaEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, kappa=0):
        self.kappa = kappa

    def fit(self, X, y):
        logger.info("Fitting KappaEncoder with shape X: %s, y: %s", X.shape, y.shape)
        try:
            self.columns = list(range(0, X.shape[1]))
            self.train_col_values = {col: X[:, col].astype(np.float32) for col in self.columns}
            self.train_target_values = y.astype(np.float32)
            # kappa may arrive as a 0-d array, a scalar, or one value per column
            if hasattr(self.kappa, 'ndim'):
                kappa_val = float(self.kappa)
            else:
                kappa_val = self.kappa

            # Expand to a per-column dict
            if isinstance(kappa_val, (int, float)):
                self.kappa = {col: kappa_val for col in self.columns}
            else:
                self.kappa = {col: kappa_val[col] for col in self.columns}
            logger.info("KappaEncoder fit complete.")
        except Exception as e:
            logger.error("Error in KappaEncoder.fit: %s", e, exc_info=True)
            raise

    def transform(self, X, y=None):
        logger.info("Transforming data with KappaEncoder, shape: %s", X.shape)
        try:
            return self.encode_numba(X.copy())
        except Exception as e:
            logger.error("Error in KappaEncoder.transform: %s", e, exc_info=True)
            raise

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def encode_numba(self, X):
        logger.debug("Encoding with Numba, shape: %s", X.shape)
        try:
            train_col_values_arr = np.array([self.train_col_values[col] for col in self.columns])
            kappa_values_arr = np.array([self.kappa[col] for col in self.columns])
            X_encoded = _encode_numba_inner(
                X,
                np.array(self.columns),
                train_col_values_arr,
                self.train_target_values,
                kappa_values_arr
            )
            return X_encoded
        except Exception as e:
            logger.error("Error in encode_numba: %s", e, exc_info=True)
            raise


class EtaRegressor(BaseEstimator, RegressorMixin):
    """Predicts each test sample as a Euclidean-distance-weighted average of the
    training targets, weighted by 1 / (1 + dist)^eta (Eq. eq:predict). The outer
    loop over test samples is parallelised across cores with `prange`; each core
    sweeps all N training points serially."""
    def __init__(self, eta=1.0):
        self.eta = float(eta) if hasattr(eta, 'ndim') else eta

    def fit(self, X, y):
        try:
            X, y = check_X_y(X, y, dtype=np.float32)
            self.X_train_ = X
            self.y_train_ = y
            return self
        except Exception as e:
            logger.error("Error in EtaRegressor.fit: %s", e, exc_info=True)
            raise

    @staticmethod
    @njit(parallel=True, cache=True)
    def _predict_chunk(X_chunk, X_train, y_train, eta):
        n_samples = X_chunk.shape[0]
        n_train = X_train.shape[0]
        n_features = X_train.shape[1]
        predictions = np.empty(n_samples, dtype=np.float32)

        for i in prange(n_samples):
            distances = np.empty(n_train, dtype=np.float32)
            for j in range(n_train):
                dist_sq = 0.0
                for k in range(n_features):
                    diff = X_chunk[i, k] - X_train[j, k]
                    dist_sq += diff * diff
                distances[j] = np.sqrt(dist_sq)

            weights = np.float32(1.0) / ((np.float32(1.0) + distances) ** eta)
            numerator = np.sum(y_train * weights)
            denominator = np.sum(weights)

            if denominator == 0:
                predictions[i] = np.mean(y_train)
            else:
                predictions[i] = numerator / denominator

        return predictions

    def predict(self, X):
        try:
            check_is_fitted(self)
            X = check_array(X, dtype=np.float32)

            if self.eta == 0:
                return np.full(X.shape[0], np.mean(self.y_train_))
            
            predictions = self._predict_chunk(X, self.X_train_, self.y_train_, np.float32(self.eta))
            return np.array(predictions)
        except Exception as e:
            logger.error("Error in EtaRegressor.predict: %s", e, exc_info=True)
            raise


class KappaEtaRegressor(BaseEstimator, RegressorMixin):
    """Algorithm 1 end to end: MinMaxScaler -> KappaEncoder -> EtaRegressor."""

    def __init__(self, kappa=2.0, eta=2.0):
        self.kappa = float(kappa) if hasattr(kappa, 'ndim') else kappa
        self.eta = float(eta) if hasattr(eta, 'ndim') else eta

    def fit(self, X, y):
        logger.info("Fitting KappaEtaRegressor with kappa=%s, eta=%s", self.kappa, self.eta)
        try:
            X, y = check_X_y(X, y, dtype=np.float32)
            self.pipeline_ = Pipeline([
                ('scaler', MinMaxScaler()),
                ('encoder', KappaEncoder(kappa=self.kappa)),
                ('regressor', EtaRegressor(eta=self.eta)),
            ])
            self.pipeline_.fit(X, y)
            logger.info("KappaEtaRegressor fit complete.")
            return self
        except Exception as e:
            logger.error("Error in KappaEtaRegressor.fit: %s", e, exc_info=True)
            raise

    def predict(self, X):
        try:
            check_is_fitted(self, 'pipeline_')
            X = check_array(X, dtype=np.float32)
            return self.pipeline_.predict(X)
        except Exception as e:
            logger.error("Error in KappaEtaRegressor.predict: %s", e, exc_info=True)
            raise