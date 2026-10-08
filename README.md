# GPU Acceleration of Distance-Based Regressors on Apple Silicon

Companion code for "On the Computational Complexity of GPU-Based Implementations of Distance-Based Regressors" research paper.

The paper models the GPU-vs-CPU speedup of a distance-based regressor mapped one
thread per output. The GPU keeps `W` threads in flight, so `M = N` scored points run
in `⌈N/W⌉` scheduling rounds, giving the saturating curve

```
S_N = S∞ · N / (N + N_sat)
```

Its two constants are recovered cheaply: the plateau `S∞ = a_c/a_g` from small `N`
runs, and the knee `N_sat = W` from the shape of the GPU timing curve alone. The
κ-η Regressor serves as the representative member of the class, accelerated here with
hand-written Metal kernels dispatched through MLX.

## What's here

| File | Role |
|------|------|
| `kappa.py` | Multi-core CPU baseline (Numba) — `KappaEncoder` / `EtaRegressor` / `KappaEtaRegressor` |
| `kappa_mlx.py` | Metal/MLX GPU implementation — `KappaEncoderMLX` / `EtaRegressorMLX` / `KappaEtaRegressorMLX` |
| `saturation_model.py` | The two fits — `fit_cpu_model`, `fit_gpu_model`, `fit_knee`, `calibrate` |
| `01_benchmark_m4.py` | CPU-vs-GPU timing sweep on an M4 -> `benchmark_data.csv` |
| `02_validate_saturation_model.ipynb` | Fits the model; produces the paper's tables and speedup figure |
| `03_thermal_analysis.ipynb` | Thermal appendix figure; confirms the benchmark was not throttled |
| `benchmark_data.csv` | The M4 timings the paper reports (`split` marks calibration `train`, `N <= 10k`, vs held-out `test`) |
| `thermals.csv` | 10 s-cadence thermal telemetry captured during the benchmark |

## Setup

Apple Silicon (macOS) is required as the Metal kernels need an Apple GPU, and MLX will not run on other hardware. Install with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Reproducing the results

`02_validate_saturation_model.ipynb` fits the model and regenerates the tables and
figure from the committed `benchmark_data.csv`. It needs no GPU, and should print:

| Quantity | Value |
|----------|-------|
| `S∞` — predicted plateau speedup | `155.9x` |
| `N_sat` — knee, measured from the GPU curve | `3806` |
| Held-out prediction error, mean / max | `2.8%` / `8.1%` |

`03_thermal_analysis.ipynb` reads `thermals.csv` and writes `thermal_temps.pdf`.

```bash
uv run jupyter lab
```

Regenerating the timing data from scratch needs an Apple GPU and takes hours of CPU
time — the CPU pass alone is ~234 s per trial at `N = 100,000`, over 12 trials:

```bash
uv run python 01_benchmark_m4.py
uv run python 01_benchmark_m4.py --smoke   # tiny sanity run instead
```
