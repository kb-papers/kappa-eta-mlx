"""
Speedup saturation model for the kappa-eta pipeline (GPU vs CPU)
================================================================

Implements Sections 6 and 7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Hardware description (Table 1, "Hardware" block)
# ---------------------------------------------------------------------------

@dataclass
class ChipSpec:
    """The hardware constants the model needs: core counts, nothing else.

    W (and hence the knee N_sat) is measured from the GPU timing curve by
    fit_knee, not predicted from a hardware datasheet.
    """
    name: str
    cpu_cores: int      # CPU cores Numba parallelises across (C_cpu)
    gpu_cores: int      # GPU compute cores (C_gpu), used only to report R = N_sat/C_gpu


CHIPS = {
    "M4": ChipSpec("M4", cpu_cores=10, gpu_cores=10),
}


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _wls(A, b, w):
    """Weighted least squares. Returns the coefficients and the weighted R^2.

    Every residual is scaled by sqrt(w), so the minimised objective is
    sum_i w_i (b_i - A_i c)^2. R^2 is computed under the same weighting, which
    is why each fit below reports an R^2 on its own terms.
    """
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    w = np.ones_like(b) if w is None else np.asarray(w, float)
    sw = np.sqrt(w)
    coeffs, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
    resid = b - A @ coeffs
    ss_res = np.sum(w * resid ** 2)
    ss_tot = np.sum(w * (b - np.average(b, weights=w)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coeffs, r2


def fit_cpu_model(N, T_cpu, P):
    """Fit the CPU timing model, Eq. eq:tcpu:  T_cpu = a_c N^2 P + g_c.

    Residuals are weighted by N^2. Only the leading coefficient a_c feeds
    S_inf = a_c/a_g, and that coefficient describes the large-N regime, so the
    weighting keeps the many cheap small N points where the fixed overhead
    g_c dominates.

    Returns {'a_c', 'g_c', 'r2'}.
    """
    N = np.asarray(N, float)
    A = np.column_stack([N ** 2 * P, np.ones_like(N)])
    (a_c, g_c), r2 = _wls(A, T_cpu, N ** 2)
    return {"a_c": a_c, "g_c": g_c, "r2": r2}


def fit_gpu_model(N, T_gpu, P):
    """Fit the GPU model in its ASYMPTOTIC form:  T_gpu = a_g N^2 P + g_g.

    Used for the plateau calibration on the cheap N <= 10,000 region only, and
    weighted by N^2 for the same reason as fit_cpu_model.

    The linear term is constrained to b_g = 0 deliberately. Over the cheap
    region the N^2 term has not yet come to dominate T_gpu, so a freely fitted
    b_g absorbs part of it and biases a_g and with it S_inf. The linear
    coefficient is instead estimated separately, over the full sweep, by
    fit_knee.

    Returns {'a_g', 'g_g', 'r2'}.
    """
    N = np.asarray(N, float)
    A = np.column_stack([N ** 2 * P, np.ones_like(N)])
    (a_g, g_g), r2 = _wls(A, T_gpu, N ** 2)
    return {"a_g": a_g, "g_g": g_g, "r2": r2}


def fit_knee(N, T_gpu, P, spec: ChipSpec):
    """Measure the knee N_sat from the GPU timing curve alone (Eq. eq:knee_fit).

    Eq. eq:tgpu_fit is a quadratic in N that is linear in its three
    coefficients, so it is fitted directly by weighted least squares

        argmin sum_i ((T_gpu(N_i) - g_g - b_g N_i - a_g P N_i^2) / T_gpu(N_i))^2

    and the knee read off as the sample counterpart of Eq. eq:nsat_deriv,
    N_sat = b_g / (a_g P).

    The 1/T_gpu in the denominator makes each residual *relative* rather than
    absolute, so every measurement contributes in proportion to its own
    magnitude. T_gpu spans three orders of magnitude across the sweep; under
    absolute residuals the fit is decided almost entirely by the largest N, and
    those points sit deep in the quadratic regime where the linear term and
    hence the knee has no influence at all.

    Fit this over the FULL sweep, not the cheap calibration subset: the
    quadratic coefficient only becomes visible once the N^2 term dominates. It
    stays inexpensive because it uses the GPU pass only.

    Returns {'a_g', 'b_g', 'g_g', 'N_sat', 'R', 'r2'} (coefficients are the
    knee-fit estimates, superscripted 'knee' in the paper).
    """
    N = np.asarray(N, float)
    T_gpu = np.asarray(T_gpu, float)
    A = np.column_stack([np.ones_like(N), N, N ** 2 * P])
    (g_g, b_g, a_g), r2 = _wls(A, T_gpu, 1.0 / T_gpu ** 2)
    N_sat = b_g / (a_g * P)
    return {"a_g": a_g, "b_g": b_g, "g_g": g_g, "r2": r2,
            "N_sat": N_sat, "R": N_sat / spec.gpu_cores}


# ---------------------------------------------------------------------------
# Calibration container + prediction
# ---------------------------------------------------------------------------

@dataclass
class ChipCalibration:
    """The calibrated model: the two fitted timing curves plus the measured knee."""
    spec: ChipSpec
    P: int
    a_c: float      # CPU quadratic coefficient (tau_cpu / C_cpu)
    g_c: float      # CPU fixed overhead
    a_g: float      # GPU quadratic coefficient (tau_gpu / W)
    g_g: float      # GPU launch overhead
    N_sat: float    # knee, from fit_knee
    r2_cpu: float
    r2_gpu: float

    @property
    def S_inf(self) -> float:
        """Plateau speedup, Eq. eq:sinf_deriv: S_inf = a_c/a_g.

        It depends only on the two leading N^2 coefficients, so it is
        knee-independent which is why it is read off the fitted slopes
        rather than by fitting Eq. eq:mm to the measured speedup ratios (the
        cheap small-N points are exactly where the neglected overheads g_c, g_g
        are not negligible; at N = 100 the GPU is in fact slower).
        """
        return self.a_c / self.a_g

    def predict_speedup(self, N):
        """The saturating speedup curve, Eq. eq:mm: S_inf N / (N + N_sat)."""
        N = np.asarray(N, float)
        return self.S_inf * N / (N + self.N_sat)


def calibrate(N, T_cpu, T_gpu, P, spec: ChipSpec, N_sat: float) -> ChipCalibration:
    """Fit both timing models on one region (the cheap subset) and attach the
    knee measured separately by fit_knee over the full GPU sweep."""
    cpu = fit_cpu_model(N, T_cpu, P)
    gpu = fit_gpu_model(N, T_gpu, P)
    return ChipCalibration(
        spec=spec, P=P,
        a_c=cpu["a_c"], g_c=cpu["g_c"],
        a_g=gpu["a_g"], g_g=gpu["g_g"],
        N_sat=float(N_sat),
        r2_cpu=cpu["r2"], r2_gpu=gpu["r2"],
    )
