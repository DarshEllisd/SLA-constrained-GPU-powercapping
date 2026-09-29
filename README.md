# SLA-Constrained GPU Power Capping via Hierarchical Roofline Characterization

Experiment harness and measured data for the paper of the same name.

A GPU's power limit is set for workloads using 100% of cores, but many kernels wait on memory much
of the time while the GPU holds its clocks high. This harness measures how much of that power can be
taken back: for a kernel with a given arithmetic intensity α and working set, it sweeps board power
caps and records the **Best Cap** — the cap that minimises energy subject to a bounded runtime
slowdown. Sweeping α across two levels of the memory hierarchy produces a cap map that a scheduler
can look up.

---

## Notation

The same symbols are used here as in the paper:

| symbol | meaning |
|---|---|
| `h ∈ {DRAM, L2}` | the memory level |
| `α = F / Q_h` | arithmetic intensity in FLOP/byte, for FP32 work `F` and bytes moved `Q_h` |
| `E_h = P_avg · t` | run energy |
| `C` | the set of requested caps |
| `Δt_h` | slowdown against the uncapped baseline |
| `τ` | SLA tolerance |
| `P*_h` | the selected cap, the **Best Cap** |
| FFMA | fused multiply–add |
| NVML | NVIDIA Management Library |

### Paper term ↔ data column

The harness predates the paper's wording, so some CSV columns carry older names. They mean the same
things:

| paper | data |
|---|---|
| Best Cap, `P*_h` | `best_cap_by_ai.csv`, columns prefixed `best_*` |
| α | `target_ai` (requested), `true_ai` (synthesised), `achieved_ai` (measured) |
| Δt | `perf_deg_pct` — **negative means faster than baseline** |
| τ | the `--deg-threshold` flag |
| `E_h` | `total_energy_j` |
| Energy Saved | `energy_saved_pct` |
| `C` | the `--power-caps` flag; `power_cap_w` per row |
| memory level `h` | the `regime` column, and the `dram/` and `l2/` directories |

"Best Cap zone" in filenames and in `hierarchical_roofline_summary.md` is the Best Cap.

---

## What the benchmark does

A generic benchmark generates any arithmetic intensity on demand by varying the bytes moved and the
compute applied to them. Each thread streams one element through the memory system and applies a
chosen number of FFMAs to it; raising that number raises α, so a single kernel covers the whole
sweep.

The memory level `h` is set by changing the size of the working set compared to the size of L2;
overflowing L2 makes it use GDDR6. The loop count is calibrated per α so every trial runs for about
the same wall time, then held fixed across the caps, so work stays constant and any shift in the
operating point is due to the cap.

Caps are applied through NVML with **no clock locks**, so the governor picks its own
voltage–frequency point under each cap. Power is sampled with `nvmlDeviceGetPowerUsage` over exactly
the timed interval, giving `(P_avg, t)` and `E_h = P_avg · t`. Runtime comes from CUDA events.

### Selection rule

For `C = {100, 110, …, 250} W` and tolerance τ:

```
P*_h(α; τ) = argmin  E_h(α, c)      s.t.   Δt_h(α, c) ≤ τ
             c ∈ C
```

with `Δt_h(α, c) = (t_h(α, c) / t_h(α, 250 W) − 1) × 100%`. Ties break on shortest runtime, then
lowest request, and τ = 8% unless stated.

This is an argmin over **energy**, not over caps. The lowest cap that meets the SLA is not the one
that uses least energy, because an aggressive cap stretches runtime enough to raise total energy.

---

## Testbed

The committed data was measured on a node with an **NVIDIA RTX 5000 Ada Generation** (AD102:
100 SMs, 12,800 FP32 cores, 64 MB L2, 32 GB GDDR6 with ECC on a 256-bit bus) under **CUDA 13.2**,
with nothing else running on the GPU.

| setting | value |
|---|---|
| Baseline / default power limit | 250 W |
| Cap sweep `C` | 250 W down to 100 W, 16 caps in 10 W steps |
| DRAM-streaming working set | 256 MB (overflows L2) |
| L2-resident working set | 16 MB (fits inside L2) |
| α sweep | 0 upward in 10 FLOP/B steps; reached 130 for DRAM, 80 for L2 |
| τ | 8% |
| Trial length | ~80 s, loop count calibrated per α |
| Replicates | 3 (`run1/`, `run2/`, `run3/`) |
| Baseline runtimes | 72.1–98.7 s |

---

## Requirements

| | |
|---|---|
| GPU | NVIDIA, compute capability **sm_89** as tested — see *Porting* below for others |
| CUDA | `nvcc` available, or set `CUDA_PATH` (defaults to `/usr/local/cuda`) |
| Python | 3.9+, with `torch`, `pynvml`, `numpy`, `pandas` |
| OS | Linux. The shell wrappers are bash; the Python harness itself is portable. |
| Privileges | **root**, because setting an NVML power limit requires it |

```bash
pip install torch pynvml numpy pandas
```

Regenerating the figure additionally needs `matplotlib`; nothing else in the analysis path does.

---

## Running

```bash
sudo bash experiment/run_experiment.sh          # builds the kernel if needed, then sweeps
```

`run_experiment.sh` uses `python3` from `PATH`; override with `PYTHON_EXEC=/path/to/venv/bin/python`.

Build and run separately:

```bash
bash experiment/compile_kernels.sh              # -> intensity_benchmark_kernels.so
sudo python3 experiment/benchmark_ai_powercaps.py
```

Check it works without touching power limits first:

```bash
python3 experiment/benchmark_ai_powercaps.py --dry-run
```

### Options

| flag | default | meaning |
|---|---|---|
| `--runs` | `3` | replicate sweeps, written to `run1/`, `run2/`, … |
| `--start-run` | `1` | first run index, for resuming |
| `--memory-target` | `both` | `both`, `dram` or `l2` — which level `h` to sweep |
| `--ai-min` / `--ai-max` / `--ai-step` | `0` / `500` / `10` | α sweep, FLOP/B |
| `--power-caps` | `250 … 100` | the cap set `C`, in W |
| `--target-duration-s` | `80` | wall time each trial is calibrated to |
| `--deg-threshold` | `8.0` | τ, in % |
| `--dram-buffer-mb` | `256` | DRAM-streaming working set |
| `--l2-buffer-mb` | `16` | L2-resident working set |
| `--warmup-iters` | `10` | warm-up passes before each timed run |
| `--saturation-count` | `8` | stop a level after this many consecutive saturated α |
| `--dry-run` | off | run kernels without setting power limits |
| `--output-dir` | `experiment/results` | where run directories are written |

`--ai-max` defaults to 500, but `--saturation-count` ends a level early: once the Best Cap has sat at
the baseline or a steady plateau for 8 consecutive steps, that level stops. This is why the committed
data reaches α = 130 for DRAM and α = 80 for L2 rather than 500.

---

## Porting to a different architecture

Four things are tied to the RTX 5000 Ada and must be changed for another GPU.

**1. Compute capability.** `experiment/compile_kernels.sh` builds with `-arch=sm_89`. Set it to the
target's capability (`sm_80` for A100, `sm_90` for H100, and so on):

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

**2. Working sets, which set the memory level `h`.** The defaults assume a 64 MB L2: 16 MB sits
comfortably inside it, 256 MB comfortably exceeds it. Both must be re-sized against the target's L2,
or the two levels stop being distinct — the whole hierarchical comparison depends on it. As a rule,
keep the L2-resident set around a quarter of L2 capacity and the DRAM-streaming set at four times it:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv    # L2 size is in the vendor datasheet
python3 experiment/benchmark_ai_powercaps.py --l2-buffer-mb 10 --dram-buffer-mb 160   # e.g. 40 MB L2
```

**3. The cap set `C`.** Requests outside the card's settable range are rejected. The **first** cap in
the list — not the largest — is taken as the uncapped baseline for Δt and Energy Saved, so list them
in descending order starting from the card's default limit, or every number will be measured against
the wrong reference:

```bash
nvidia-smi -q -d POWER | grep -E "Default Power Limit|Min Power Limit|Max Power Limit"
python3 experiment/benchmark_ai_powercaps.py --power-caps 300 290 280 270 260 250 240 230 220 210 200
```

**4. Trial length.** Power is sampled at 1 Hz, so trials need to be long enough for the average to be
meaningful. 80 s is comfortable; do not go far below ~30 s.

τ (`--deg-threshold`) is a policy choice, not an architectural one — it stays whatever slowdown you
are willing to accept.

Expect the results themselves to differ. The Best Cap map is specific to a device; another card needs
its own sweep.

---

## Data layout

```
results/
  run1/ run2/ run3/                      three replicate sweeps
    dram/                                h = DRAM, 256 MB working set
      ai_0/ ai_10/ … ai_130/
        powercap_100.csv … powercap_250.csv    1 Hz telemetry
      raw_sweep_data.csv                 one row per (α, cap)
      best_cap_by_ai.csv              Best Cap per α
    l2/                                  h = L2, 16 MB working set
      ai_0/ … ai_80/
      raw_sweep_data.csv
      best_cap_by_ai.csv
    hierarchical_roofline_summary.md     per-run summary
```

**`raw_sweep_data.csv`** is where analysis should start. One row per (α, cap):

| column | meaning |
|---|---|
| `regime` | memory level `h`: `DRAM` or `L2` |
| `target_ai`, `true_ai`, `achieved_ai` | requested, synthesised and measured α |
| `kernel_fma` | FFMAs per element, the knob that sets α |
| `power_cap_w` | requested cap — **not** delivered power |
| `runtime_s` | kernel time from CUDA events |
| `avg_power_w`, `peak_power_w` | NVML board power over the timed interval |
| `avg_clock_mhz`, `avg_temp_c` | mean core clock and temperature |
| `total_energy_j` | `E_h = avg_power_w × runtime_s` |
| `perf_deg_pct` | `Δt` against the 250 W baseline; negative means faster |
| `energy_saved_j`, `energy_saved_pct` | Energy Saved against the 250 W baseline |
| `achieved_bw_gbs`, `achieved_tflops` | measured bandwidth and compute |

**`best_cap_by_ai.csv`** gives the Best Cap per α. It is produced by exactly the rule above, so it
can be re-derived from `raw_sweep_data.csv`.

### Re-deriving the Best Cap

```python
import csv, statistics as st

RUNS, TAU = ["run1", "run2", "run3"], 8.0

def load(h, run):
    path = f"results/{run}/{h}/raw_sweep_data.csv"
    return {(r["target_ai"], int(r["power_cap_w"])): r for r in csv.DictReader(open(path))}

def best_cap(runs, alpha):
    """argmin mean E_h, subject to dt <= TAU in every replicate."""
    caps = sorted({c for (a, c) in runs[0] if a == alpha})
    feasible = {c: st.mean([float(r[(alpha, c)]["total_energy_j"]) for r in runs])
                for c in caps
                if max(float(r[(alpha, c)]["perf_deg_pct"]) for r in runs) <= TAU}
    return min(feasible, key=feasible.get)

runs = [load("dram", r) for r in RUNS]
print(best_cap(runs, "10.0"))     # -> 150
```

---

## Regenerating the figure

`make_fig_savings.py` reads `results/` and plots Energy Saved against α for both memory levels, with
±1σ error bars over the three replicates:

```bash
python3 make_fig_savings.py        # -> fig_savings.png (400 dpi)
```

It re-applies the selection rule to `raw_sweep_data.csv` rather than reading precomputed values, so
the plot cannot drift from the measured data. It also prints the plotted numbers, which makes the
figure diffable against any other analysis:

```
DRAM  0:32.10+-0.54, 10:36.41+-1.02, 20:30.46+-0.04, ...
L2    0:17.57+-0.03, 10:10.12+-0.16, 20:9.70+-0.05, ...
```

Paths resolve relative to the script, so it runs from any working directory.

---

## Caveats

These match the paper's Limitations.

- `power_cap_w` is the **requested** limit, not delivered power. At α = 0 every request between
  100 W and 140 W draws a mean 143.2–143.4 W; the cap is not enforced below that floor.
- Clocks are **not** locked. The GPU's own voltage–frequency governor stays active under every cap,
  which is deliberate — locking clocks measures a machine that does not exist in deployment.
- `E_h` comes from NVML's power reading over the timed interval, not from an external meter.
- These are microbenchmarks, run on a single isolated GPU. A shared GPU, or a real application whose
  behaviour changes as it runs, was not tested.
- The Best Cap values are particular to this GPU; another card needs its own sweep.

---

## Repository layout

```
experiment/
  benchmark_ai_powercaps.py       sweep harness
  intensity_benchmark_kernels.cu  parameterised CUDA kernel
  compile_kernels.sh              builds the .so (sm_89)
  run_experiment.sh               compile + run wrapper
make_fig_savings.py               plots Energy Saved vs α from results/
results/                          measured data, three replicate sweeps
```

`intensity_benchmark_kernels.so` and `fig_savings.png` are generated and deliberately not committed;
`compile_kernels.sh` and `make_fig_savings.py` reproduce them.
