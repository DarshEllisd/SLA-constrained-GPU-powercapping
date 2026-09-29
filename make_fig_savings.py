"""Generate Fig. 2 of abstract.tex: energy saved vs arithmetic intensity.

Reads results/run{1,2,3}/{dram,l2}/raw_sweep_data.csv directly, so the figure is
always consistent with the data in the paper. Re-run after any sweep change:

    python make_fig_savings.py

Output: fig_savings.png (400 dpi, one IEEE column wide).
"""
import csv
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUTPUT = os.path.join(HERE, "fig_savings.png")
RUNS = ["run1", "run2", "run3"]
TAU = 8.0                      # SLA tolerance, matches Eq. 1
AIS = [float(x) for x in range(0, 90, 10)]

# Okabe-Ito colourblind-safe pair; validated (CVD dE 21.9 protan, 31.2 normal).
# Line style and marker shape repeat the distinction so identity survives
# greyscale printing.
STYLE = {
    "DRAM": dict(color="#0072B2", marker="o", linestyle="-", label="DRAM streaming"),
    "L2":   dict(color="#D55E00", marker="s", linestyle="--", label="L2-resident"),
}
INK = "#222222"        # text/axis ink - never the series colour
GRID = "#D9D9D9"


def load(regime, run):
    path = f"{RESULTS}/{run}/{regime}/raw_sweep_data.csv"
    with open(path) as fh:
        return {(r["target_ai"], int(r["power_cap_w"])): r for r in csv.DictReader(fh)}


def best_cap(runs, ai):
    """argmin over mean energy, subject to dt <= TAU in every replicate (Eq. 1)."""
    caps = sorted({c for (a, c) in runs[0] if a == ai})
    feasible = {
        c: st.mean([float(r[(ai, c)]["total_energy_j"]) for r in runs])
        for c in caps
        if max(float(r[(ai, c)]["perf_deg_pct"]) for r in runs) <= TAU
    }
    return min(feasible, key=feasible.get)


def series(regime):
    runs = [load(regime, r) for r in RUNS]
    means, sigmas = [], []
    for ai in AIS:
        key = f"{ai:.1f}"
        cap = best_cap(runs, key)
        saved = [float(r[(key, cap)]["energy_saved_pct"]) for r in runs]
        means.append(st.mean(saved))
        sigmas.append(st.stdev(saved) if len(saved) > 1 else 0.0)
    return means, sigmas


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 7,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
    })

    # Height is tuned so the figure renders ~4.1 cm tall at \columnwidth,
    # which is what the two-page limit allows.
    fig, ax = plt.subplots(figsize=(3.40, 1.55))

    for regime in ("DRAM", "L2"):
        means, sigmas = series(regime.lower())
        ax.errorbar(
            AIS, means, yerr=sigmas,
            linewidth=1.1, markersize=3.0, markeredgewidth=0,
            elinewidth=0.7, capsize=1.6, capthick=0.7,
            zorder=3, **STYLE[regime],
        )

    ax.set_xlabel("Arithmetic intensity (FLOP/B)")
    ax.set_ylabel("Energy saved (%)")
    ax.set_xlim(-3, 83)
    ax.set_ylim(-2, 41)
    ax.set_xticks(range(0, 90, 20))
    ax.set_yticks(range(0, 41, 10))

    # recessive chrome: horizontal grid only, no top/right spines
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.tick_params(colors=INK, width=0.6, length=2.5, pad=2)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)

    leg = ax.legend(loc="upper right", frameon=False, handlelength=1.8,
                    borderpad=0.2, labelspacing=0.25, handletextpad=0.5)
    for text in leg.get_texts():
        text.set_color(INK)

    fig.savefig(OUTPUT, dpi=400, bbox_inches="tight", pad_inches=0.01)
    print("wrote fig_savings.png")

    # echo the plotted values so they can be diffed against the paper
    for regime in ("DRAM", "L2"):
        means, sigmas = series(regime.lower())
        pts = ", ".join(f"{a:.0f}:{m:.2f}+-{s:.2f}" for a, m, s in zip(AIS, means, sigmas))
        print(f"  {regime:5} {pts}")


if __name__ == "__main__":
    main()
