#!/usr/bin/env python3
"""Analyse a bilevel_result.json file produced by app-bilevel.

Generates:
  - convergence.png: outer objective value g vs iteration
  - weights.png: stacked area (evolution of observable weights)
  - params_traj.png: parameter trajectories over iterations
  - params_std_tail.png: bar chart of per-parameter std dev over tail iterations
  - eobs_fairness.png: variance & coefficient of variation of per-observable errors over iterations
  - summary.txt: textual statistics (best iteration, fairness improvement, etc.)

Assumptions / Notes:
  * The JSON was produced by apprentice.bilevel.OuterBilevel.to_jsonable.
  * Observable order: recovered from insertion order of the first entry's e_obs dict.
    Python >=3.7 preserves insertion order for dict; the writer used that order to build e_obs.
  * Weight vectors are assumed in that same order (true for current implementation).

Usage:
  python examples/analyse_bilevel.py bilevel_result.json -o analysis_out

Dependencies: numpy, matplotlib
"""
import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
    raise SystemExit("matplotlib is required for this analysis script. Install it and retry.") from e


@dataclass
class HistoryEntry:
    weights: np.ndarray
    params: np.ndarray
    g: float
    e_obs: Dict[str, float]
    score_obs: Dict[str, float] | None
    meta: Dict[str, Any]


def load_history(path: str) -> tuple[Dict[str, Any], List[HistoryEntry], List[str]]:
    with open(path) as f:
        data = json.load(f)
    hist_raw = data["history"]
    if not hist_raw:
        raise ValueError("History is empty in file: " + path)
    # Observable order from first e_obs dict insertion order
    first_keys = list(hist_raw[0]["e_obs"].keys())
    history = [
        HistoryEntry(
            weights=np.array(h["weights"], dtype=float),
            params=np.array(h["params"], dtype=float),
            g=float(h["g"]),
            e_obs=h["e_obs"],
            score_obs=h.get("score_obs"),
            meta=h.get("meta", {}),
        )
        for h in hist_raw
    ]
    return data, history, first_keys


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def plot_convergence(history: List[HistoryEntry], outdir: str):
    gvals = [h.g for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(gvals, marker='o', lw=1)
    plt.xlabel('Iteration')
    plt.ylabel('g (outer objective)')
    plt.title('Convergence of outer objective')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'convergence.png'), dpi=150)
    plt.close()


def plot_weights(history: List[HistoryEntry], obs_order: List[str], outdir: str):
    W = np.vstack([h.weights for h in history])  # shape (T, n_obs)
    t = np.arange(len(history))
    plt.figure(figsize=(7, 4))
    plt.stackplot(t, W.T, labels=obs_order)
    plt.xlabel('Iteration')
    plt.ylabel('Weight share')
    plt.title('Observable weight evolution (stack plot)')
    if len(obs_order) <= 15:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'weights.png'), dpi=150)
    plt.close()


def plot_params(history: List[HistoryEntry], outdir: str, tail_frac: float):
    P = np.vstack([h.params for h in history])  # (T, n_params)
    T, n_params = P.shape
    t = np.arange(T)
    # Trajectories
    plt.figure(figsize=(7, 4))
    for i in range(n_params):
        plt.plot(t, P[:, i], label=f'p{i}')
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value')
    plt.title('Parameter trajectories')
    if n_params <= 15:
        plt.legend(ncol=2, fontsize='x-small')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'params_traj.png'), dpi=150)
    plt.close()
    # Std dev over tail
    tail_start = int((1 - tail_frac) * T)
    tail = P[tail_start:]
    std_tail = np.std(tail, axis=0, ddof=1) if tail.shape[0] > 1 else np.zeros(n_params)
    plt.figure(figsize=(7, 4))
    plt.bar(np.arange(n_params), std_tail)
    plt.xlabel('Parameter index')
    plt.ylabel('Std dev (tail)')
    plt.title(f'Parameter stability (last {tail_frac*100:.0f}% iterations)')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'params_std_tail.png'), dpi=150)
    plt.close()
    return std_tail, tail_start


def plot_fairness(history: List[HistoryEntry], obs_order: List[str], outdir: str):
    # Build matrix of e_obs with consistent ordering
    E = np.vstack([[h.e_obs[k] for k in obs_order] for h in history])  # (T, n_obs)
    var_across = np.var(E, axis=1, ddof=0)
    mean_across = np.mean(E, axis=1)
    # Coefficient of variation (std/mean) handling mean=0 case
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = np.sqrt(var_across) / mean_across
        cv[~np.isfinite(cv)] = 0
    t = np.arange(len(history))
    plt.figure(figsize=(6, 4))
    plt.plot(t, var_across, label='Variance of e_obs')
    plt.plot(t, cv, label='Coeff. variation (std/mean)')
    plt.xlabel('Iteration')
    plt.title('Per-observable fairness metrics')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'eobs_fairness.png'), dpi=150)
    plt.close()
    return var_across, cv, mean_across


def write_summary(path: str, data: Dict[str, Any], history: List[HistoryEntry], obs_order: List[str],
                  std_tail: np.ndarray, tail_start: int, var_across: np.ndarray, cv: np.ndarray, mean_across: np.ndarray):
    best_idx = int(np.argmin([h.g for h in history]))
    best = history[best_idx]
    with open(path, 'w') as f:
        f.write(f"Objective: {data['objective']}\n")
        if data['objective'] == 'portfolio':
            f.write(f"lambda_var: {data.get('lambda_var')}\n")
        f.write(f"Iterations: {len(history)}\n")
        f.write(f"Best g: {best.g:.6g} at iteration {best_idx}\n")
        # Parameter names if present
        pnames = data.get('pnames')
        if pnames and len(pnames) == len(best.params):
            f.write("Best parameters (name:value):\n")
            for pn, pv in zip(pnames, best.params):
                f.write(f"  {pn}: {pv:.8g}\n")
        else:
            f.write("Best parameter vector:\n  [" + ", ".join(f"{v:.8g}" for v in best.params) + "]\n")
        f.write("Best weights (observable:value):\n")
        for k, w in zip(obs_order, best.weights):
            f.write(f"  {k}: {w:.6f}\n")
        f.write("Per-observable errors at best:\n")
        for k in obs_order:
            f.write(f"  {k}: {best.e_obs[k]:.6g}\n")
        f.write("\nFairness progression (variance of e_obs):\n")
        f.write(f"  start: {var_across[0]:.6g}\n")
        f.write(f"  end:   {var_across[-1]:.6g}\n")
        if var_across[0] > 0:
            f.write(f"  reduction factor: {var_across[-1]/var_across[0]:.3f}\n")
        f.write("\nCoefficient of variation (start -> end): ")
        f.write(f"{cv[0]:.3f} -> {cv[-1]:.3f}\n")
        f.write("\nMean e_obs (start -> end): ")
        f.write(f"{mean_across[0]:.6g} -> {mean_across[-1]:.6g}\n")
        f.write("\nParameter stability (std dev over tail starting at iteration {}):\n".format(tail_start))
        for i, s in enumerate(std_tail):
            f.write(f"  p{i}: {s:.6g}\n")
        f.write("\nInner objective (raw) at best: {}\n".format(best.meta.get('inner_fun', 'n/a')))
        f.write("\nDone.\n")


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Analyse bilevel optimization result JSON.")
    ap.add_argument('json', help='Path to bilevel_result.json')
    ap.add_argument('-o', '--outdir', default='bilevel_analysis', help='Output directory for plots & summary')
    ap.add_argument('--tail-frac', type=float, default=0.3, help='Fraction of last iterations for stability stats (default 0.3)')
    args = ap.parse_args()

    ensure_dir(args.outdir)
    data, history, obs_order = load_history(args.json)
    plot_convergence(history, args.outdir)
    plot_weights(history, obs_order, args.outdir)
    std_tail, tail_start = plot_params(history, args.outdir, tail_frac=args.tail_frac)
    var_across, cv, mean_across = plot_fairness(history, obs_order, args.outdir)
    write_summary(os.path.join(args.outdir, 'summary.txt'), data, history, obs_order,
                  std_tail, tail_start, var_across, cv, mean_across)
    print("Analysis complete. See directory:", args.outdir)


if __name__ == '__main__':  # pragma: no cover
    main()
