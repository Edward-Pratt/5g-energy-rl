#!/usr/bin/env python3
"""
plot_rollout.py  –  Visualise and compare PPO / DQN training & eval logs.

Usage examples
--------------
# Single run
python plot_rollout.py --files rollout_ppo.csv --labels PPO

# Compare PPO vs DQN training
python plot_rollout.py --files rollout_ppo.csv rollout_dqn.csv --labels PPO DQN

# Compare with custom output directory
python plot_rollout.py --files rollout_ppo.csv rollout_dqn.csv \
                       --labels PPO DQN --outdir my_plots
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Colour palette (consistent across all plots) ─────────────────────────────
PALETTE = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & preparation
# ─────────────────────────────────────────────────────────────────────────────

# Columns that every log file should have (filled with NaN if absent)
REQUIRED_COLS = [
    "thr", "delay", "jitter", "loss", "numUe",
    "stepEnergyJ", "rew", "act_m", "txPowerDbm", "sinr",
    # DQN-only
    "act_idx", "epsilon", "q_loss",
]


def load(path: str) -> pd.DataFrame:
    """Load a CSV log, normalise column names, and add helper columns."""
    df = pd.read_csv(path)

    # Normalise timestamp column
    if "ts" not in df.columns and "t" in df.columns:
        df = df.rename(columns={"t": "ts"})

    # Ensure episode / step columns exist
    if "episode" not in df.columns:
        df["episode"] = 0
    if "step" not in df.columns:
        df["step"] = np.arange(len(df), dtype=int)

    # Fill missing required columns with NaN
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # Unified continuous action column: prefer act_m (both agents log this)
    df["act"] = df["act_m"]

    # Sort and build a global step axis
    df = df.sort_values(["episode", "step"]).reset_index(drop=True)
    horizon = int(df.groupby("episode")["step"].max().max() + 1)
    df["gstep"] = df["episode"] * horizon + df["step"]

    return df


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived engineering metrics."""
    d = df.copy()
    d["thr_mbps"]      = d["thr"] / 1e6
    d["delay_ms"]      = d["delay"] * 1e3
    d["jitter_ms"]     = d["jitter"] * 1e3
    denom              = (1.0 - d["loss"]).clip(lower=1e-6)
    d["offered_mbps"]  = d["thr_mbps"] / denom
    d["delivery_ratio"]= (1.0 - d["loss"]).clip(0.0, 1.0)
    energy             = d["stepEnergyJ"].replace(0, np.nan)
    d["eff_mbps_per_j"]= d["thr_mbps"] / energy
    return d


def episode_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per episode with aggregate stats."""
    g = df.groupby("episode", as_index=False)
    out = g.agg(
        steps       = ("step",        "max"),
        ep_return   = ("rew",         "sum"),
        mean_rew    = ("rew",         "mean"),
        mean_thr    = ("thr",         "mean"),
        mean_delay  = ("delay",       "mean"),
        mean_jitter = ("jitter",      "mean"),
        mean_loss   = ("loss",        "mean"),
        mean_energy = ("stepEnergyJ", "mean"),
    )
    out["steps"] += 1
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, outdir: str, name: str, dpi: int = 150):
    fig.savefig(os.path.join(outdir, name), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _rolling(series: pd.Series, window: int = 10) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


def _has_data(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and not df[col].isna().all()


def _agent_label(label: str, df: pd.DataFrame) -> str:
    """Append agent type hint to label if detectable."""
    if _has_data(df, "epsilon"):
        return f"{label} (DQN)"
    return f"{label} (PPO)"


# ─────────────────────────────────────────────────────────────────────────────
# Plot groups
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(dfs, labels, outdir):
    """
    The most important group: shows whether the agent is actually improving.
    Produces 3 figures:
      1. Episode return (raw + rolling average)
      2. Step-level reward with 100-step moving average
      3. Reward distribution: early vs late episodes
    """
    summaries = [episode_summary(df) for df in dfs]

    # ── 1. Episode return ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for (s, df, lab), color in zip(zip(summaries, dfs, labels), PALETTE):
        lab = _agent_label(lab, df)
        ax.plot(s["episode"], s["ep_return"],
                color=color, alpha=0.3, linewidth=1)
        ax.plot(s["episode"], _rolling(s["ep_return"], 10),
                color=color, linewidth=2.5, label=lab)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return")
    ax.set_title("Episode Return (faint = raw, solid = 10-ep rolling average)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, outdir, "lc_episode_return.png")

    # ── 2. Step-level reward moving average ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    for (df, lab), color in zip(zip(dfs, labels), PALETTE):
        lab = _agent_label(lab, df)
        ma = df["rew"].rolling(100, min_periods=1).mean()
        ax.plot(df["gstep"], ma, color=color, linewidth=2, label=lab)
    ax.set_xlabel("Global step")
    ax.set_ylabel("Reward (100-step rolling avg)")
    ax.set_title("Step-Level Reward Trend")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, outdir, "lc_reward_moving_avg.png")

    # ── 3. Reward distribution: first 10 % vs last 10 % of episodes ──────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for (df, lab), color in zip(zip(dfs, labels), PALETTE):
        lab = _agent_label(lab, df)
        n_ep      = df["episode"].max() + 1
        early_cut = max(1, int(n_ep * 0.10))
        late_cut  = max(1, int(n_ep * 0.90))
        early = df[df["episode"] <  early_cut]["rew"].dropna()
        late  = df[df["episode"] >= late_cut ]["rew"].dropna()
        kw = dict(bins=30, alpha=0.6, color=color, label=lab)
        if len(early): axes[0].hist(early, **kw)
        if len(late):  axes[1].hist(late,  **kw)
    for ax, title in zip(axes, ["Early episodes (first 10 %)", "Late episodes (last 10 %)"]):
        ax.set_title(title)
        ax.set_xlabel("Reward per step")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Reward Distribution – shift right = agent is learning")
    _save(fig, outdir, "lc_reward_distribution.png")


def plot_agent_specific(dfs, labels, outdir):
    """
    DQN: epsilon decay + Q-loss.
    PPO: these columns are NaN and the plot is skipped gracefully.
    """
    # ── Epsilon decay (DQN only) ──────────────────────────────────────────────
    dqn_pairs = [(df, lab) for df, lab in zip(dfs, labels) if _has_data(df, "epsilon")]
    if dqn_pairs:
        fig, ax = plt.subplots(figsize=(10, 3))
        for (df, lab), color in zip(dqn_pairs, PALETTE):
            ax.plot(df["gstep"], pd.to_numeric(df["epsilon"], errors="coerce"),
                    color=color, linewidth=1.5, label=f"{lab} ε")
        ax.set_xlabel("Global step")
        ax.set_ylabel("Epsilon")
        ax.set_title("DQN Exploration: Epsilon Decay")
        ax.legend()
        ax.grid(True, alpha=0.3)
        _save(fig, outdir, "dqn_epsilon.png")

    # ── Q-loss (DQN only) ─────────────────────────────────────────────────────
    dqn_loss_pairs = [(df, lab) for df, lab in zip(dfs, labels) if _has_data(df, "q_loss")]
    if dqn_loss_pairs:
        fig, ax = plt.subplots(figsize=(10, 3))
        for (df, lab), color in zip(dqn_loss_pairs, PALETTE):
            ql = pd.to_numeric(df["q_loss"], errors="coerce")
            ax.plot(df["gstep"], ql.rolling(50, min_periods=1).mean(),
                    color=color, linewidth=1.5, label=f"{lab} Q-loss (50-step avg)")
        ax.set_xlabel("Global step")
        ax.set_ylabel("Smooth L1 loss")
        ax.set_title("DQN Q-Loss (should trend downward)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        _save(fig, outdir, "dqn_q_loss.png")

    # ── DQN discrete action index distribution ────────────────────────────────
    dqn_act_pairs = [(df, lab) for df, lab in zip(dfs, labels) if _has_data(df, "act_idx")]
    if dqn_act_pairs:
        fig, ax = plt.subplots(figsize=(8, 4))
        for (df, lab), color in zip(dqn_act_pairs, PALETTE):
            counts = df["act_idx"].value_counts().sort_index()
            counts = counts / counts.sum()
            ax.bar(counts.index + list(PALETTE).index(color) * 0.3,
                   counts.values, width=0.3, alpha=0.7, color=color, label=lab)
        ax.set_xlabel("Action index")
        ax.set_ylabel("Fraction of steps")
        ax.set_title("DQN Discrete Action Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        _save(fig, outdir, "dqn_action_index_distribution.png")


def plot_qos(dfs, labels, outdir, warmup_drop=0):
    """Delay / jitter / loss over global steps."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for (df, lab), color in zip(zip(dfs, labels), PALETTE):
        d = derive(df[df["gstep"] >= warmup_drop] if warmup_drop else df)
        lab = _agent_label(lab, df)
        axes[0].plot(d["gstep"], d["delay_ms"],  color=color, label=lab, linewidth=1)
        axes[1].plot(d["gstep"], d["jitter_ms"], color=color, label=lab, linewidth=1)
        axes[2].plot(d["gstep"], d["loss"],      color=color, label=lab, linewidth=1)
    labels_ax = ["Delay (ms)", "Jitter (ms)", "Packet loss"]
    for ax, ylabel in zip(axes, labels_ax):
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Global step")
    fig.suptitle("QoS Metrics")
    _save(fig, outdir, "qos_over_time.png")


def plot_throughput_energy(dfs, labels, outdir, warmup_drop=0):
    """Throughput / energy / efficiency over global steps."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for (df, lab), color in zip(zip(dfs, labels), PALETTE):
        d = derive(df[df["gstep"] >= warmup_drop] if warmup_drop else df)
        lab = _agent_label(lab, df)
        axes[0].plot(d["gstep"], d["thr_mbps"],       color=color, label=lab, linewidth=1)
        axes[1].plot(d["gstep"], d["stepEnergyJ"],    color=color, label=lab, linewidth=1)
        axes[2].plot(d["gstep"], d["eff_mbps_per_j"], color=color, label=lab, linewidth=1)
    labels_ax = ["Throughput (Mbps)", "Step energy (J)", "Efficiency (Mbps/J)"]
    for ax, ylabel in zip(axes, labels_ax):
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Global step")
    fig.suptitle("Throughput, Energy and Efficiency")
    _save(fig, outdir, "throughput_energy_efficiency.png")


def plot_actions(dfs, labels, outdir, warmup_drop=0):
    """Continuous action value (act_m) and TX power over time."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for (df, lab), color in zip(zip(dfs, labels), PALETTE):
        d = df[df["gstep"] >= warmup_drop] if warmup_drop else df
        lab = _agent_label(lab, df)
        axes[0].plot(d["gstep"], d["act"],        color=color, label=lab, linewidth=1)
        axes[1].plot(d["gstep"], d["txPowerDbm"], color=color, label=lab, linewidth=1)
    axes[0].set_ylabel("Action (m)")
    axes[1].set_ylabel("TX Power (dBm)")
    axes[-1].set_xlabel("Global step")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Actions and TX Power")
    _save(fig, outdir, "actions_txpower.png")

    # Action vs throughput scatter (useful to see which action region pays off)
    fig, ax = plt.subplots(figsize=(7, 4))
    for (df, lab), color in zip(zip(dfs, labels), PALETTE):
        d = derive(df)
        lab = _agent_label(lab, df)
        ax.scatter(d["act"], d["thr_mbps"], s=6, alpha=0.4, color=color, label=lab)
    ax.set_xlabel("Action (m)")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("Action vs Throughput")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, outdir, "action_vs_throughput.png")


def plot_per_episode_summary(dfs, labels, outdir):
    """Mean throughput / delay / energy per episode."""
    summaries = [episode_summary(df) for df in dfs]
    metrics = [
        ("mean_thr",    "Mean throughput"),
        ("mean_delay",  "Mean delay (s)"),
        ("mean_energy", "Mean step energy (J)"),
    ]
    for col, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(9, 3))
        for (s, df, lab), color in zip(zip(summaries, dfs, labels), PALETTE):
            lab = _agent_label(lab, df)
            ax.plot(s["episode"], s[col], marker="o", markersize=3,
                    color=color, label=lab, linewidth=1.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fname = f"per_episode_{col}.png"
        _save(fig, outdir, fname)


def plot_single_episode(df, label, outdir):
    """Detailed plots when a single episode is passed."""
    d = derive(df)
    x = d["step"]
    color = PALETTE[0]

    # Throughput / energy / reward
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(x, d["thr_mbps"],    color=color)
    axes[1].plot(x, d["stepEnergyJ"], color=color)
    axes[2].plot(x, d["rew"],         color=color)
    for ax, ylabel in zip(axes, ["Throughput (Mbps)", "Step energy (J)", "Reward"]):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Step")
    fig.suptitle(f"{label} – Single episode")
    _save(fig, outdir, "single_thr_energy_reward.png")

    # QoS
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(x, d["delay_ms"],  color=color)
    axes[1].plot(x, d["jitter_ms"], color=color)
    axes[2].plot(x, d["loss"],      color=color)
    for ax, ylabel in zip(axes, ["Delay (ms)", "Jitter (ms)", "Loss"]):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Step")
    fig.suptitle(f"{label} – QoS")
    _save(fig, outdir, "single_qos.png")

    # Cumulative reward and energy side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(x, d["rew"].cumsum(), color=color)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cumulative reward")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(x, d["stepEnergyJ"].fillna(0).cumsum(), color=color)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Cumulative energy (J)")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"{label} – Cumulative metrics")
    _save(fig, outdir, "single_cumulative.png")

    # Action + TX power
    fig, ax1 = plt.subplots(figsize=(9, 3))
    ax1.plot(x, d["act"], color=color, label="Action (m)")
    ax1.set_ylabel("Action (m)")
    ax1.set_xlabel("Step")
    if _has_data(d, "txPowerDbm"):
        ax2 = ax1.twinx()
        ax2.plot(x, d["txPowerDbm"], color=PALETTE[1], linestyle="--", label="TX Power (dBm)")
        ax2.set_ylabel("TX Power (dBm)")
        lines  = ax1.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels)
    ax1.grid(True, alpha=0.3)
    fig.suptitle(f"{label} – Actions")
    _save(fig, outdir, "single_action_txpower.png")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Plot and compare PPO / DQN training logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--files",        nargs="+", required=True,
                    help="One or more CSV log files")
    ap.add_argument("--labels",       nargs="*",
                    help="Display labels (same count as --files)")
    ap.add_argument("--outdir",       default="plots",
                    help="Directory to write PNG files into")
    ap.add_argument("--warmup_drop",  type=int, default=0,
                    help="Skip first N global steps from timeseries plots")
    args = ap.parse_args()

    # ── Resolve labels ────────────────────────────────────────────────────────
    if args.labels and len(args.labels) == len(args.files):
        labels = args.labels
    else:
        labels = [os.path.splitext(os.path.basename(f))[0] for f in args.files]

    os.makedirs(args.outdir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    dfs = [load(f) for f in args.files]
    for path, df, lab in zip(args.files, dfs, labels):
        agent_type = "DQN" if _has_data(df, "epsilon") else "PPO"
        print(f"  Loaded {path!r} → {len(df):,} rows, "
              f"{df['episode'].nunique()} episodes [{agent_type}] as '{lab}'")

    # ── Decide single-episode vs multi-episode mode ───────────────────────────
    single = len(dfs) == 1 and dfs[0]["episode"].nunique() == 1

    if single:
        plot_single_episode(dfs[0], labels[0], args.outdir)
        print("\nSingle-episode plots written.")
    else:
        plot_learning_curves(dfs, labels, args.outdir)
        plot_agent_specific(dfs, labels, args.outdir)
        plot_qos(dfs, labels, args.outdir, args.warmup_drop)
        plot_throughput_energy(dfs, labels, args.outdir, args.warmup_drop)
        plot_actions(dfs, labels, args.outdir, args.warmup_drop)
        plot_per_episode_summary(dfs, labels, args.outdir)

        print("\n⭐ Key training plots:")
        print(f"   {args.outdir}/lc_episode_return.png")
        print(f"   {args.outdir}/lc_reward_moving_avg.png")
        print(f"   {args.outdir}/lc_reward_distribution.png")
        print(f"   {args.outdir}/dqn_epsilon.png  (DQN only)")
        print(f"   {args.outdir}/dqn_q_loss.png   (DQN only)")

    print(f"\n✅ All plots written to: {args.outdir}/")


if __name__ == "__main__":
    main()