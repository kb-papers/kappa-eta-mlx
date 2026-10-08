#!/usr/bin/env python
"""01_benchmark_m4.py — M4 CPU-vs-GPU timing benchmark (data generator).

Produces the timing data behind Section 7 (Methodology) of the paper: a sweep of
the kappa-eta pipeline over 100 <= N <= 100,000. Written to benchmark_data.csv.
This script only generates data; the two fits and the plateau prediction live downstream in
02_validate_saturation_model.ipynb.

Pipeline: KappaEncoder -> EtaRegressor, timed on both backends:
  - CPU  — Numba JIT   (kappa.py)
  - GPU  — Metal / MLX (kappa_mlx.py)

A trial is one complete, self-contained sweep: a full CPU pass followed by a
full GPU pass over every N. Repeats are never taken back-to-back at one N — those share cache and thermal state.
Measuring each N once per trial, with an entire sweep in between, spaces the samples out
in time, and notebook 02 collates them into the per N mean and 95% CI the
paper reports. The data itself (random_state=42) is identical across trials,
so only the timing varies.

Usage
-----
  uv run python 01_benchmark_m4.py                  # default trials -> benchmark_data.csv
  uv run python 01_benchmark_m4.py --trials 5       # trials this invocation
  uv run python 01_benchmark_m4.py --trials 1 --append   # accumulate across processes
  uv run python 01_benchmark_m4.py --smoke          # tiny check -> benchmark_data_smoke.csv
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline

from kappa import KappaEncoder, EtaRegressor            # CPU (Numba)
from kappa_mlx import KappaEncoderMLX, EtaRegressorMLX   # GPU (Metal / MLX)

# ---------------------------------------------------------------------------
# Configuration. The hardware constants only label the output CSV — no model
# fitting happens here, and nothing downstream predicts the knee from them
# (notebook 02 measures it from the GPU timing curve).
# ---------------------------------------------------------------------------
# -- Pipeline / data parameters --
N_FEATURES    = 10
N_INFORMATIVE = 2
KAPPA         = 6.0
ETA           = 5.0

# -- Hardware label (Apple M4) --
PROCESSOR = 'M4'
CPU_CORES = 10
GPU_CORES = 10
TG        = 256          # threadgroup size

# -- Sweep control --
CALIB_MAX_N = 10500      # N <= this is the cheap calibration subset (split='train');
                         # larger N is the held-out plateau (split='test')

ROW_COUNTS_FULL  = [100, 500, 1000, 2500, 5000, 7500, 10000, 20000, 30000,
                    40000, 60000, 80000, 100000]
ROW_COUNTS_SMOKE = [100, 500, 1000]

DEFAULT_TRIALS_FULL  = 12    # the paper reports 12 independent trials
DEFAULT_TRIALS_SMOKE = 1


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------
def build_cpu_pipeline(kappa, eta):
    return Pipeline([('encoder', KappaEncoder(kappa=kappa)),
                     ('regressor', EtaRegressor(eta=eta))])


def build_gpu_pipeline(kappa, eta):
    return Pipeline([('encoder', KappaEncoderMLX(kappa=kappa)),
                     ('regressor', EtaRegressorMLX(eta=eta))])


def benchmark_pipeline(pipeline, X, y):
    """Single fit+predict wall-clock time, in seconds (one measurement).

    Independence comes from the trial structure (see run_sweep), not from
    repeating this call — we never average back-to-back repeats here.
    """
    t0 = time.perf_counter(); pipeline.fit(X, y);      t_fit  = time.perf_counter() - t0
    t0 = time.perf_counter(); _ = pipeline.predict(X); t_pred = time.perf_counter() - t0
    return t_fit + t_pred


def make_datasets(row_counts):
    """Fixed (deterministic) datasets, shared across all trials."""
    datasets = {}
    for n in row_counts:
        X, y = make_regression(n_samples=n, n_features=N_FEATURES,
                               n_informative=N_INFORMATIVE, random_state=42, noise=0.1)
        datasets[n] = (X.astype(np.float32), y.astype(np.float32))
    return datasets


def warmup():
    """Warm up both backends so the first timed run isn't penalised by one-time costs.

    GPU: power-state ramp-up. CPU: Numba JIT compilation — `_encode_numba_inner`
    and `kappa_impute_numba` (kappa.py) have no `cache=True`, so they compile fresh
    every process. Numba compiles per type *signature* (float32, N_FEATURES cols),
    not per array size, so one throwaway pass primes every N that follows.
    """
    print("Warming up CPU (Numba JIT compile)...", flush=True)
    Xw = np.random.rand(256, N_FEATURES).astype(np.float32)
    yw = np.random.rand(256).astype(np.float32)
    p = build_cpu_pipeline(KAPPA, ETA); p.fit(Xw, yw); _ = p.predict(Xw)

    print("Warming up GPU...", flush=True)
    for wn in [100, 1000, 3000]:
        Xw = np.random.rand(wn, N_FEATURES).astype(np.float32)
        yw = np.random.rand(wn).astype(np.float32)
        p = build_gpu_pipeline(5.0, 1.0); p.fit(Xw, yw); _ = p.predict(Xw)
    print("Warm-up done.", flush=True)


def run_sweep(row_counts, datasets, trial):
    """One independent trial: CPU pass then GPU pass. Returns a tidy DataFrame
    tagged with the `trial` index (one row per N)."""
    print(f"=== CPU pass (trial {trial}) ===", flush=True)
    t_cpu = {}
    for n in row_counts:
        X, y = datasets[n]
        t_cpu[n] = benchmark_pipeline(build_cpu_pipeline(KAPPA, ETA), X, y)
        print(f"  N={n:>7,}: {t_cpu[n]:8.4f}s", flush=True)

    print(f"=== GPU pass (trial {trial}) ===", flush=True)
    t_gpu = {}
    for n in row_counts:
        X, y = datasets[n]
        t_gpu[n] = benchmark_pipeline(build_gpu_pipeline(KAPPA, ETA), X, y)
        print(f"  N={n:>7,}: {t_gpu[n]:8.4f}s", flush=True)

    rows = [{
        'processor': PROCESSOR, 'cpu_cores': CPU_CORES, 'gpu_cores': GPU_CORES,
        'threadgroup': TG, 'n_features': N_FEATURES, 'kappa': KAPPA, 'eta': ETA,
        'split': 'train' if n <= CALIB_MAX_N else 'test', 'trial': trial, 'N': n,
        'T_cpu_s': t_cpu[n], 'T_gpu_s': t_gpu[n], 'speedup': t_cpu[n] / t_gpu[n],
    } for n in row_counts]
    return pd.DataFrame(rows)


def summarize(df):
    """Per-N mean +/- std across trials — a quick sanity check printed at the end."""
    return (df.groupby(['split', 'N'])
              .agg(T_cpu_mean=('T_cpu_s', 'mean'), T_cpu_std=('T_cpu_s', 'std'),
                   T_gpu_mean=('T_gpu_s', 'mean'), T_gpu_std=('T_gpu_s', 'std'),
                   speedup_mean=('speedup', 'mean'), speedup_std=('speedup', 'std'),
                   n_trials=('speedup', 'size'))
              .reset_index()
              .sort_values('N'))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="M4 CPU-vs-GPU timing benchmark (data generator).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-n', '--trials', type=int, default=None,
                   help="Number of independent sweeps to run THIS invocation (each "
                        "becomes one trial). Default: 12 (1 with --smoke).")
    p.add_argument('--append', action='store_true',
                   help="Append to the existing CSV and continue the trial index, "
                        "instead of overwriting. Run repeatedly to accumulate "
                        "process-independent trials.")
    p.add_argument('--smoke', action='store_true',
                   help="Tiny fast sweep to benchmark_data_smoke.csv (leaves the "
                        "real benchmark_data.csv untouched).")
    p.add_argument('-o', '--output', default=None,
                   help="Output CSV path (default: benchmark_data.csv, or "
                        "benchmark_data_smoke.csv with --smoke).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    row_counts = ROW_COUNTS_SMOKE if args.smoke else ROW_COUNTS_FULL
    n_trials = (args.trials if args.trials is not None
                else (DEFAULT_TRIALS_SMOKE if args.smoke else DEFAULT_TRIALS_FULL))
    if n_trials < 1:
        raise SystemExit(f"--trials must be >= 1 (got {n_trials})")
    output = Path(args.output) if args.output else Path(
        'benchmark_data_smoke.csv' if args.smoke else 'benchmark_data.csv')

    # When appending, continue the trial index past whatever is already on disk.
    existing = None
    start_trial = 0
    if args.append and output.exists():
        existing = pd.read_csv(output)
        if 'trial' in existing.columns and len(existing):
            start_trial = int(existing['trial'].max()) + 1

    mode = 'SMOKE TEST' if args.smoke else 'FULL RUN'
    print(f"{mode} | {PROCESSOR} C_cpu={CPU_CORES} C_gpu={GPU_CORES} TG={TG}")
    print(f"Row counts: {row_counts}")
    action = 'append to' if args.append else 'overwrite'
    print(f"Trials this run: {n_trials} (trial ids {start_trial}..{start_trial + n_trials - 1})"
          f"  ->  {action} {output}\n", flush=True)

    warmup()
    datasets = make_datasets(row_counts)   # fixed data, shared across trials

    trials = []
    for i in range(n_trials):
        trial = start_trial + i
        print(f"\n########## TRIAL {i + 1}/{n_trials} (id {trial}) ##########", flush=True)
        trials.append(run_sweep(row_counts, datasets, trial))

    df = pd.concat(trials, ignore_index=True)
    if existing is not None:
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(output, index=False)

    total_trials = df['trial'].nunique() if 'trial' in df.columns else n_trials
    print(f"\nWrote {output} ({len(df)} rows; {total_trials} total trials x "
          f"{len(row_counts)} N)")
    print("\nPer-N summary (mean +/- std across trials):")
    print(summarize(df).to_string(index=False))


if __name__ == '__main__':
    main()
