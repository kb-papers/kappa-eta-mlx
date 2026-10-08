"""
kappa-eta pipeline — Apple Silicon GPU implementation (Metal via MLX)
=====================================================================

GPU-accelerated implementation of the two-stage kappa-eta regression pipeline
(KappaEncoder -> EtaRegressor), Section 5 of the accompanying paper. Both
stages are O(N^2 * P): every test sample is compared against all N training
samples. We map that work onto the Apple GPU with hand-written Metal kernels
launched through `mx.fast.metal_kernel`.

Key design choices (the optimizations that make the GPU path fast):

Metal kernel design:
- One GPU thread per output element. The encoder launches N_test x P threads (one
  per (sample, feature)); the regressor launches N_test threads (one per sample),
  each looping serially over the N training points.
- 8x loop unrolling in the encoder's inner training loop to expose
  instruction-level parallelism, with a scalar remainder loop for the tail.
- float4-vectorized Euclidean distance in the regressor (4 features per step via
  metal::dot), with a scalar remainder.
- Fused distance + weight computation, and flat inner loops (no inner chunking), to
  keep register pressure low.

Python-side design:
- SIMD-aligned threadgroups: sizes are rounded to a multiple of 32 (the Apple GPU
  SIMD width) so no execution slots are wasted (`_align_threadgroup`).
- Lazy evaluation respected: MLX arrays are accumulated across batches and
  evaluated ONCE at the end. Calling `mx.eval()` inside the batch loop would stall
  the GPU pipeline on every batch.
- Single GPU->CPU transfer: one `np.array()` after `mx.concatenate`, rather than a
  copy per batch (unified memory still requires explicit materialization).

All paths use float32 throughout (required by the Metal kernels).
"""

import numpy as np
import mlx.core as mx
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import math

# --- Constants for Apple Silicon GPU ---
SIMD_WIDTH = 32  # Apple GPU SIMD width (all Apple Silicon)
DEFAULT_THREADGROUP = 256  # Launch threadgroup size (a multiple of SIMD_WIDTH)

# ============================================================================
# ENCODER KERNEL
# ============================================================================
# One thread per (test sample, feature). Each thread scans all N training rows
# for its feature column, accumulating kappa-weighted target values. The inner
# loop is 8x-unrolled for instruction-level parallelism, with a scalar remainder.
ENCODER_KERNEL = """
    uint gid = thread_position_in_grid.x;

    if (gid >= total_ops_val) return;

    uint sample_idx = gid / n_features_val;
    uint col_idx = gid % n_features_val;

    float x_target = x_test[sample_idx * n_features_val + col_idx];
    float kappa_val = kappa_values[col_idx];

    // Early exit: kappa=0 means uniform weights → just return mean
    if (metal::abs(kappa_val) < 1e-9f) {
        encoded_out[gid] = mean_y_val;
        return;
    }

    float neg_kappa = -kappa_val;
    float weighted_sum_y = 0.0f;
    float sum_weights = 0.0f;

    // Main loop, 8x unrolled for instruction-level parallelism
    uint unroll_limit = (n_train_samples_val / 8) * 8;

    for (uint i = 0; i < unroll_limit; i += 8) {
        // Load 8 training values for this feature column
        float x0 = x_train[i * n_features_val + col_idx];
        float x1 = x_train[(i+1) * n_features_val + col_idx];
        float x2 = x_train[(i+2) * n_features_val + col_idx];
        float x3 = x_train[(i+3) * n_features_val + col_idx];
        float x4 = x_train[(i+4) * n_features_val + col_idx];
        float x5 = x_train[(i+5) * n_features_val + col_idx];
        float x6 = x_train[(i+6) * n_features_val + col_idx];
        float x7 = x_train[(i+7) * n_features_val + col_idx];

        // Fused distance + weight: abs → add 1 → pow
        float w0 = metal::fast::pow(1.0f + metal::abs(x_target - x0), neg_kappa);
        float w1 = metal::fast::pow(1.0f + metal::abs(x_target - x1), neg_kappa);
        float w2 = metal::fast::pow(1.0f + metal::abs(x_target - x2), neg_kappa);
        float w3 = metal::fast::pow(1.0f + metal::abs(x_target - x3), neg_kappa);
        float w4 = metal::fast::pow(1.0f + metal::abs(x_target - x4), neg_kappa);
        float w5 = metal::fast::pow(1.0f + metal::abs(x_target - x5), neg_kappa);
        float w6 = metal::fast::pow(1.0f + metal::abs(x_target - x6), neg_kappa);
        float w7 = metal::fast::pow(1.0f + metal::abs(x_target - x7), neg_kappa);

        // Accumulate weights and weighted y in two groups to reduce dependency chains
        float w_sum_a = w0 + w1 + w2 + w3;
        float w_sum_b = w4 + w5 + w6 + w7;
        sum_weights += w_sum_a + w_sum_b;

        float wy_a = w0*y_train[i] + w1*y_train[i+1] + w2*y_train[i+2] + w3*y_train[i+3];
        float wy_b = w4*y_train[i+4] + w5*y_train[i+5] + w6*y_train[i+6] + w7*y_train[i+7];
        weighted_sum_y += wy_a + wy_b;
    }

    // Remainder loop (no unrolling needed for < 8 elements)
    for (uint i = unroll_limit; i < n_train_samples_val; ++i) {
        float x_val = x_train[i * n_features_val + col_idx];
        float w = metal::fast::pow(1.0f + metal::abs(x_target - x_val), neg_kappa);
        sum_weights += w;
        weighted_sum_y += w * y_train[i];
    }

    encoded_out[gid] = (sum_weights > eps_val) ? (weighted_sum_y / sum_weights) : mean_y_val;
"""

# ============================================================================
# REGRESSOR KERNEL
# ============================================================================
# One thread per test sample. Each thread loops over all N training points,
# computing the Euclidean distance (float4-vectorized over features) and an
# eta-weighted average of the training targets.
REGRESSOR_KERNEL = """
    uint gid = thread_position_in_grid.x;

    if (gid >= curr_batch_size_val) return;

    float neg_eta = -eta_val;
    float weighted_sum_y = 0.0f;
    float sum_weights = 0.0f;

    device const float* test_sample = x_batch + gid * n_feat_val;

    for (uint i = 0; i < n_train_samp_val; ++i) {
        device const float* train_sample = x_train + i * n_feat_val;

        float dist_sq = 0.0f;

        // Vectorized distance: process 4 features at a time using float4
        uint vec4_limit = (n_feat_val / 4) * 4;
        for (uint j = 0; j < vec4_limit; j += 4) {
            float4 t_vec = float4(test_sample[j], test_sample[j+1],
                                  test_sample[j+2], test_sample[j+3]);
            float4 r_vec = float4(train_sample[j], train_sample[j+1],
                                  train_sample[j+2], train_sample[j+3]);
            float4 diff = t_vec - r_vec;
            dist_sq += metal::dot(diff, diff);
        }

        // Scalar remainder
        for (uint j = vec4_limit; j < n_feat_val; ++j) {
            float diff = test_sample[j] - train_sample[j];
            dist_sq += diff * diff;
        }

        float distance = metal::fast::sqrt(dist_sq + eps_dist_val);
        float weight = metal::fast::pow(1.0f + distance, neg_eta);

        sum_weights += weight;
        weighted_sum_y += weight * y_train[i];
    }

    predictions_out[gid] = (sum_weights > eps_div_val) ?
                           (weighted_sum_y / sum_weights) : mean_y_val;
"""


def _align_threadgroup(n_threads, simd_width=SIMD_WIDTH, max_tg=DEFAULT_THREADGROUP):
    """
    Round the threadgroup size up to a multiple of the SIMD width.

    Apple GPUs execute in SIMD groups of 32 threads. Threadgroup sizes
    that aren't multiples of 32 leave the extra lanes idle.
    
    Args:
        n_threads: Total number of threads needed
        simd_width: SIMD width of the GPU (32 for Apple Silicon)
        max_tg: Maximum threadgroup size to use
    
    Returns:
        Threadgroup size as a multiple of simd_width
    """
    if n_threads <= simd_width:
        return simd_width  # Minimum one SIMD group
    # Round up to nearest multiple of simd_width, capped at max_tg
    aligned = min(((n_threads + simd_width - 1) // simd_width) * simd_width, max_tg)
    return aligned


class KappaEncoderMLX(BaseEstimator, TransformerMixin):
    """
    Kappa Encoder — Metal GPU acceleration.

    Target encoding: each feature value is replaced by a kappa-weighted average of
    the training targets, weighted by 1 / (1 + |x_t - x_i|)^kappa. Launches one GPU
    thread per (sample, feature). Uses SIMD-aligned threadgroups, defers all
    mx.eval() to the end, and does a single GPU->CPU transfer.

    Args:
        kappa: Decay parameter (scalar or dict per feature)
        batch_size: Samples per GPU batch (None = auto 8192)
        epsilon: Numerical stability constant
    """
    
    def __init__(self, kappa=0.0, batch_size=None, epsilon=1e-9):
        if isinstance(kappa, dict):
            self.kappa = {k: float(v) for k, v in kappa.items()}
        else:
            self.kappa = float(kappa)
        self.batch_size = batch_size
        self.epsilon = float(epsilon)
        
        self._metal_kernel = mx.fast.metal_kernel(
            name="encoder",
            input_names=[
                "x_test", "x_train", "y_train", "kappa_values",
                "mean_y_val", "eps_val", "n_features_val",
                "n_train_samples_val", "total_ops_val", "batch_size_val"
            ],
            output_names=["encoded_out"],
            source=ENCODER_KERNEL
        )

    def _get_batch_size(self, n_samples):
        """Return batch size: user-specified or default."""
        if self.batch_size is not None:
            return min(self.batch_size, n_samples)
        return min(8192, n_samples)

    def fit(self, X, y):
        X, y = check_X_y(X, y, ensure_2d=True, dtype=np.float32, y_numeric=True)
        self.n_features_in_ = X.shape[1]
        self.columns_ = list(range(self.n_features_in_))

        self.X_train_ = mx.array(X.astype(np.float32, order='C'), dtype=mx.float32)
        self.y_train_ = mx.array(y.astype(np.float32), dtype=mx.float32)

        # Build per-feature kappa array
        is_single = isinstance(self.kappa, (int, float))
        kappa_list = [float(self.kappa) if is_single else float(self.kappa.get(col, 0.0))
                      for col in self.columns_]
        self.kappa_values_ = mx.array(kappa_list, dtype=mx.float32)
        self.mean_y_train_ = mx.mean(self.y_train_)

        # Single eval for all fit-time arrays
        mx.eval(self.X_train_, self.y_train_, self.kappa_values_, self.mean_y_train_)
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        X = check_array(X, ensure_2d=True, dtype=np.float32, allow_nd=False)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        n_samples = X.shape[0]
        if n_samples == 0:
            return np.empty((0, self.n_features_in_), dtype=np.float32)

        batch_size = self._get_batch_size(n_samples)
        n_batches = math.ceil(n_samples / batch_size)

        # Pre-compute scalar constants (once, outside loop)
        eps_mx = mx.array(self.epsilon, dtype=mx.float32)
        n_features_mx = mx.array(self.n_features_in_, dtype=mx.uint32)
        n_train_mx = mx.array(self.X_train_.shape[0], dtype=mx.uint32)

        # Accumulate MLX arrays — defer GPU→CPU copy to the end
        mlx_results = []

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            current_batch_size = end_idx - start_idx
            n_operations = current_batch_size * self.n_features_in_

            X_batch_mx = mx.array(X[start_idx:end_idx], dtype=mx.float32)
            batch_size_mx = mx.array(current_batch_size, dtype=mx.uint32)
            total_ops_mx = mx.array(n_operations, dtype=mx.uint32)

            # SIMD-aligned threadgroup
            tg_size = _align_threadgroup(n_operations)

            kernel_outputs = self._metal_kernel(
                inputs=[
                    X_batch_mx, self.X_train_, self.y_train_, self.kappa_values_,
                    self.mean_y_train_, eps_mx, n_features_mx, n_train_mx,
                    total_ops_mx, batch_size_mx
                ],
                grid=(n_operations, 1, 1),
                threadgroup=(tg_size, 1, 1),
                output_shapes=[(n_operations,)],
                output_dtypes=[mx.float32]
            )

            encoded_batch = mx.reshape(kernel_outputs[0], (current_batch_size, self.n_features_in_))
            mlx_results.append(encoded_batch)

        # Single eval + CPU transfer at the end
        if len(mlx_results) == 1:
            mx.eval(mlx_results[0])
            return np.array(mlx_results[0])
        else:
            result = mx.concatenate(mlx_results, axis=0)
            mx.eval(result)
            return np.array(result)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class EtaRegressorMLX(BaseEstimator, RegressorMixin):
    """
    Eta Regressor — Metal GPU acceleration.

    Predicts each test sample as a Euclidean-distance-weighted average of the
    training targets, weighted by 1 / (1 + dist)^eta. Launches one GPU thread per
    test sample; each thread loops over all N training points (float4-vectorized
    distance). Uses SIMD-aligned threadgroups and a single deferred GPU->CPU copy.

    Args:
        eta: Decay parameter for distance weighting
        batch_size: Samples per GPU batch (None = auto 4096)
        epsilon_dist: Distance computation epsilon
        epsilon_div: Division stability epsilon
    """
    
    def __init__(self, eta=0.0, batch_size=None, epsilon_dist=1e-9, epsilon_div=1e-9):
        self.eta = float(eta)
        self.batch_size = batch_size
        self.epsilon_dist = float(epsilon_dist)
        self.epsilon_div = float(epsilon_div)

        self._metal_kernel = mx.fast.metal_kernel(
            name="regressor",
            input_names=[
                "x_batch", "x_train", "y_train",
                "eta_val", "mean_y_val",
                "eps_dist_val", "eps_div_val",
                "n_feat_val", "n_train_samp_val",
                "curr_batch_size_val"
            ],
            output_names=["predictions_out"],
            source=REGRESSOR_KERNEL
        )

    def _get_batch_size(self, n_samples):
        """Return batch size: user-specified or default."""
        if self.batch_size is not None:
            return min(self.batch_size, n_samples)
        return min(4096, n_samples)

    def fit(self, X, y):
        X, y = check_X_y(X, y, ensure_2d=True, dtype=np.float32, y_numeric=True)
        self.n_features_in_ = X.shape[1]

        self.X_train_ = mx.array(X.astype(np.float32, order='C'), dtype=mx.float32)
        self.y_train_ = mx.array(y.astype(np.float32), dtype=mx.float32)
        self.mean_y_train_ = mx.mean(self.y_train_)

        mx.eval(self.X_train_, self.y_train_, self.mean_y_train_)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        X_np = check_array(X, ensure_2d=True, dtype=np.float32, allow_nd=False)

        if X_np.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_np.shape[1]}")

        n_samples = X_np.shape[0]
        if n_samples == 0:
            return np.array([], dtype=np.float32)

        if self.eta == 0.0:
            return np.full((n_samples,), float(self.mean_y_train_.item()), dtype=np.float32)

        batch_size = self._get_batch_size(n_samples)
        n_batches = math.ceil(n_samples / batch_size)

        # Pre-compute scalar constants (once, outside loop)
        eta_mx = mx.array(self.eta, dtype=mx.float32)
        mean_y_mx = self.mean_y_train_
        eps_dist_mx = mx.array(self.epsilon_dist, dtype=mx.float32)
        eps_div_mx = mx.array(self.epsilon_div, dtype=mx.float32)
        n_features_mx = mx.array(self.n_features_in_, dtype=mx.uint32)
        n_train_mx = mx.array(self.X_train_.shape[0], dtype=mx.uint32)

        # Accumulate MLX arrays — defer GPU→CPU copy
        mlx_results = []

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            current_batch_size = end_idx - start_idx

            X_batch_mx = mx.array(X_np[start_idx:end_idx], dtype=mx.float32)
            batch_size_mx = mx.array(current_batch_size, dtype=mx.uint32)

            # SIMD-aligned threadgroup
            tg_size = _align_threadgroup(current_batch_size)

            kernel_outputs = self._metal_kernel(
                inputs=[
                    X_batch_mx, self.X_train_, self.y_train_,
                    eta_mx, mean_y_mx,
                    eps_dist_mx, eps_div_mx,
                    n_features_mx, n_train_mx,
                    batch_size_mx
                ],
                grid=(current_batch_size, 1, 1),
                threadgroup=(tg_size, 1, 1),
                output_shapes=[(current_batch_size,)],
                output_dtypes=[mx.float32]
            )

            mlx_results.append(kernel_outputs[0])

        # Single eval + CPU transfer
        if len(mlx_results) == 1:
            mx.eval(mlx_results[0])
            return np.array(mlx_results[0])
        else:
            result = mx.concatenate(mlx_results, axis=0)
            mx.eval(result)
            return np.array(result)


# ============================================================================
# Combined Pipeline
# ============================================================================

class KappaEtaRegressorMLX(BaseEstimator, RegressorMixin):
    """Combined Kappa-Eta pipeline: MinMaxScaler -> Metal encoder -> Metal regressor."""

    def __init__(self, kappa=2.0, eta=2.0):
        self.kappa = float(kappa) if isinstance(kappa, (int, float)) else kappa
        self.eta = float(eta) if isinstance(eta, (int, float)) else eta

    def fit(self, X, y):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler

        X, y = check_X_y(X, y, dtype=np.float32)

        self.pipeline_ = Pipeline([
            ('scaler', MinMaxScaler()),
            ('encoder', KappaEncoderMLX(kappa=self.kappa)),
            ('regressor', EtaRegressorMLX(eta=self.eta)),
        ])
        self.pipeline_.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, 'pipeline_')
        X = check_array(X, dtype=np.float32)
        return self.pipeline_.predict(X)
