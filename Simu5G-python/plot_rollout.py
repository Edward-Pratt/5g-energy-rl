#!/usr/bin/env python3
"""
plot_rollout.py  –  Visualise and compare PPO / DQN training & eval logs.

Usage examples
--------------
# Single run
python plot_rollout.py --files rollout_ppo.csv --labels PPO

# Compare PPO vs DQN
python plot_rollout.py --files rollout_ppo.csv rollout_dqn.csv --labels PPO DQN

# Variable-traffic scenario (marks phase boundaries at 20 s and 40 s)
python plot_rollout.py --files rollout_ppo.csv --labels PPO --var_traffic

# Skip noisy warm-up steps
python plot_rollout.py --files rollout_ppo.csv --labels PPO --warmup_drop 200
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Global style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":      120,
    "font.size":       16,
    "axes.titlesize":  18,
    "axes.labelsize":  16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 18,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "legend.framealpha": 0.7,
})
PALETTE = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Tick interval used in the simulator (seconds) — used to label the x-axis
# in real time when plotting a single episode.
TICK_INTERVAL_S = 0.1

# Phase boundaries for VarTraffic scenarios (in simulation seconds).
PHASE_BOUNDARIES_S = [20.0, 40.0]

# Required columns — filled with NaN if absent so every function can assume
# they exist.
REQUIRED_COLS = [
    "thr", "delay", "jitter", "loss", "numUe",
    "stepEnergyJ", "rew", "ep_return", "act_m", "txPowerDbm", "sinr",
    "act_idx", "epsilon", "q_loss",
]


# ── Data loading ───────────────────────────────────────────────────────────────

def load(path: str) -> pd.DataFrame:
    """
    Load a CSV log produced by serverppo.py or serverdqn.py.
    Normalises column names and adds helper columns.
    """
    df = pd.read_csv(path)

    if "ts" not in df.columns and "t" in df.columns:
        df = df.rename(columns={"t": "ts"})

    if "episode" not in df.columns:
        df["episode"] = 0
    if "step" not in df.columns:
        df["step"] = np.arange(len(df), dtype=int)

    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df = df.sort_values(["episode", "step"]).reset_index(drop=True)

    # FIX #1: build gstep from a per-episode step counter that resets to 0,
    # rather than assuming every episode has the same length.
    # This is correct even when episodes have different step counts (e.g.
    # early termination or varying sim-time-limit).
    ep_lengths = df.groupby("episode")["step"].transform("count")
    ep_start   = df.groupby("episode").cumcount()
    episode_offsets = (
        df.groupby("episode")["step"]
        .count()
        .shift(fill_value=0)
        .cumsum()
    )
    df["gstep"] = df["episode"].map(episode_offsets) + ep_start

    # Real-time axis (seconds within episode, for single-episode plots)
    df["sim_time_s"] = df["step"] * TICK_INTERVAL_S

    return df


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived engineering metrics used across multiple plots."""
    d = df.copy()
    d["thr_mbps"]       = d["thr"] / 1e6
    d["delay_ms"]       = d["delay"] * 1e3
    d["jitter_ms"]      = d["jitter"] * 1e3
    d["delivery_ratio"] = (1.0 - d["loss"]).clip(0.0, 1.0)
    energy              = d["stepEnergyJ"].replace(0, np.nan)
    d["eff_mbps_per_j"] = d["thr_mbps"] / energy
    # Normalised QoS score mirrors the reward function sign convention:
    # higher is better.  Used in the Pareto plot.
    d["qos_score"] = (
            d["delivery_ratio"]
            - d["delay"].clip(0) * 2.0
            - d["jitter"].clip(0) * 0.5
    )
    return d


def episode_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per episode with aggregate statistics."""
    g   = df.groupby("episode", as_index=False)
    out = g.agg(
        steps        = ("step",        "count"),
        ep_return    = ("rew",         "sum"),
        mean_rew     = ("rew",         "mean"),
        mean_thr     = ("thr",         "mean"),
        mean_delay   = ("delay",       "mean"),
        mean_jitter  = ("jitter",      "mean"),
        mean_loss    = ("loss",        "mean"),
        mean_energy  = ("stepEnergyJ", "mean"),
        mean_sinr    = ("sinr",        "mean"),
        mean_txpower = ("txPowerDbm",  "mean"),
    )
    return out


# ── Small helpers ──────────────────────────────────────────────────────────────

def _save(fig, outdir: str, name: str):
    root, ext = os.path.splitext(name)
    if ext.lower() != ".svg":
        name = f"{root}.svg"
    path = os.path.join(outdir, name)
    fig.savefig(path, bbox_inches="tight", format="svg")
    plt.close(fig)
    print(f"  Wrote {path}")


def _roll(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


def _has(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and not df[col].isna().all()


def _agent_type(df: pd.DataFrame) -> str:
    return "DQN" if _has(df, "epsilon") else "PPO"


def _label(lab: str, df: pd.DataFrame) -> str:
    return f"{lab} ({_agent_type(df)})"


def _phase_lines(ax, alpha: float = 0.5):
    """Draw vertical dashed lines at VarTraffic phase boundaries."""
    for t in PHASE_BOUNDARIES_S:
        step = int(t / TICK_INTERVAL_S)
        ax.axvline(step, color="grey", linestyle="--", linewidth=1,
                   alpha=alpha, label="_nolegend_")


def _phase_lines_time(ax, alpha: float = 0.5):
    """Phase boundary lines on a real-time (seconds) x-axis."""
    for t in PHASE_BOUNDARIES_S:
        ax.axvline(t, color="grey", linestyle="--", linewidth=1,
                   alpha=alpha, label="_nolegend_")


# ── Plot group 1: Learning curves ──────────────────────────────────────────────

def plot_learning_curves(dfs, labels, outdir):
    """
    The most important group.  Shows whether the agent is actually improving.

    Plots produced:
      lc_episode_return.svg   – cumulative return per episode
      lc_reward_trend.svg     – 100-step rolling mean of step reward
      lc_reward_distribution.svg – early vs late episode reward histograms
    """
    summaries = [episode_summary(df) for df in dfs]

    # ── Episode return ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for s, df, lab, color in zip(summaries, dfs, labels, PALETTE):
        lab = _label(lab, df)
        ax.plot(s["episode"], s["ep_return"],
                color=color, alpha=0.25, linewidth=1)
        ax.plot(s["episode"], _roll(s["ep_return"], 10),
                color=color, linewidth=2.5, label=lab)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return")
    ax.set_title("Episode Return\n"
                 "(faint = raw, solid = 10-episode rolling average)")
    ax.legend()
    _save(fig, outdir, "lc_episode_return.svg")

    # ── Step-level reward rolling mean ────────────────────────────────────────
    # FIX #5/#6: always show a rolling mean, never raw step noise.
    fig, ax = plt.subplots(figsize=(12, 4))
    for df, lab, color in zip(dfs, labels, PALETTE):
        lab = _label(lab, df)
        ax.plot(df["gstep"], _roll(df["rew"], 100),
                color=color, linewidth=2, label=lab)
    ax.set_xlabel("Global step")
    ax.set_ylabel("Reward (100-step rolling avg)")
    ax.set_title("Step-Level Reward Trend")
    ax.legend()
    _save(fig, outdir, "lc_reward_trend.svg")

    # ── Reward distribution: first 10 % vs last 10 % of episodes ─────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for df, lab, color in zip(dfs, labels, PALETTE):
        lab   = _label(lab, df)
        n_ep  = df["episode"].max() + 1
        early = df[df["episode"] <  max(1, int(n_ep * 0.10))]["rew"].dropna()
        late  = df[df["episode"] >= max(1, int(n_ep * 0.90))]["rew"].dropna()
        kw = dict(bins=40, alpha=0.6, color=color, label=lab, density=True)
        if len(early): axes[0].hist(early, **kw)
        if len(late):  axes[1].hist(late,  **kw)
    for ax, title in zip(axes, [
        "Early episodes (first 10 %)", "Late episodes (last 10 %)"
    ]):
        ax.set_title(title)
        ax.set_xlabel("Reward per step")
        ax.set_ylabel("Density")
        ax.legend()
    fig.suptitle("Reward Distribution  ·  shift right = agent is learning")
    _save(fig, outdir, "lc_reward_distribution.svg")


# ── Plot group 2: QoS metrics ──────────────────────────────────────────────────

def plot_qos(dfs, labels, outdir, warmup_drop: int = 0, var_traffic: bool = False):
    """
    Delay / jitter / packet loss per episode (mean ± std band).

    Using per-episode aggregates rather than raw step data avoids the
    30,000-point smear problem (FIX #6).  The band shows variability.

    Delay and loss share a single chart (dual y-axis); jitter is plotted
    beneath on its own axes.
    """
    summaries = [episode_summary(df) for df in dfs]

    # ── Delay + Loss on the same axes (dual y-axis) ───────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax_delay = axes[0]
    ax_loss  = ax_delay.twinx()

    delay_lines = []
    loss_lines  = []

    for s, df, lab, color in zip(summaries, dfs, labels, PALETTE):
        agent_lab = _label(lab, df)

        # Delay — solid line, left y-axis
        ln = ax_delay.plot(
            s["episode"], s["mean_delay"],
            color=color, linewidth=1.5,
            linestyle="-",
            label=f"{agent_lab} – delay",
        )
        delay_lines.extend(ln)

        # Loss — dashed line, right y-axis (uses a slightly lighter shade)
        ln = ax_loss.plot(
            s["episode"], s["mean_loss"],
            color=color, linewidth=1.5,
            linestyle="--",
            label=f"{agent_lab} – loss",
        )
        loss_lines.extend(ln)

    # ITU-T reference lines
    ax_delay.axhline(0.150, color="red",    linestyle=":", linewidth=1,
                     label="ITU-T 150 ms target")
    ax_loss.axhline( 0.010, color="orange", linestyle=":", linewidth=1,
                     label="ITU-T 1 % target")

    ax_delay.set_ylabel("Mean delay (s)",       color="black")
    ax_loss.set_ylabel( "Mean packet loss",     color="black")
    ax_delay.set_title("Delay & Packet Loss per Episode\n"
                       "(solid = delay / left axis, dashed = loss / right axis)")

    # Merge both axes' legend handles into one legend on ax_delay
    all_lines  = delay_lines + loss_lines
    all_labels = [l.get_label() for l in all_lines]
    # Also add the threshold reference lines
    all_lines  += [ax_delay.get_lines()[-1], ax_loss.get_lines()[-1]]
    all_labels += ["ITU-T 150 ms (delay)", "ITU-T 1 % (loss)"]
    ax_delay.legend(all_lines, all_labels, fontsize=16, loc="upper right")

    # Disable the auto-legend on the twin axis to avoid duplication
    ax_loss.legend().set_visible(False) if ax_loss.get_legend() else None

    # ── Jitter on its own subplot ─────────────────────────────────────────────
    ax_jitter = axes[1]
    for s, df, lab, color in zip(summaries, dfs, labels, PALETTE):
        lab = _label(lab, df)
        ax_jitter.plot(s["episode"], s["mean_jitter"],
                       color=color, linewidth=1.5, label=lab)
    ax_jitter.axhline(0.005, color="red", linestyle=":", linewidth=1,
                      label="ITU-T 5 ms target")
    ax_jitter.set_ylabel("Mean jitter (s)")
    ax_jitter.set_xlabel("Episode")
    ax_jitter.legend(fontsize=16)
    ax_jitter.set_title("Jitter per Episode")

    fig.suptitle("QoS Metrics per Episode  ·  (red/orange dotted = ITU-T G.131 targets)",
                 y=1.01)
    fig.tight_layout()
    _save(fig, outdir, "qos_per_episode.svg")

    # ── Single-episode QoS timeseries (first df only, for debugging) ──────────
    if len(dfs) == 1:
        df  = derive(dfs[0])
        ep0 = df[df["episode"] == df["episode"].max()]

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(ep0["sim_time_s"], _roll(ep0["delay_ms"],  5), color=PALETTE[0])
        axes[0].axhline(150,  color="red",    linestyle=":", linewidth=1)
        axes[1].plot(ep0["sim_time_s"], _roll(ep0["jitter_ms"], 5), color=PALETTE[0])
        axes[1].axhline(5,    color="red",    linestyle=":", linewidth=1)
        axes[2].plot(ep0["sim_time_s"], _roll(ep0["loss"],      5), color=PALETTE[0])
        axes[2].axhline(0.01, color="red",    linestyle=":", linewidth=1)
        if var_traffic:
            for ax in axes:
                _phase_lines_time(ax)
        for ax, ylabel in zip(axes, ["Delay (ms)", "Jitter (ms)", "Packet loss"]):
            ax.set_ylabel(ylabel)
        axes[-1].set_xlabel("Simulation time (s)")
        fig.suptitle(f"{labels[0]} – QoS in final episode\n"
                     f"(red dotted = ITU-T target"
                     + (", grey dashed = traffic phase boundary)" if var_traffic else ")"))
        _save(fig, outdir, "qos_final_episode.svg")


# ── Plot group 3: Power control ────────────────────────────────────────────────

def plot_power_control(dfs, labels, outdir, var_traffic: bool = False):
    """
    The core objective of the project: how does the agent choose TX power
    over time, and does it correlate with SINR and energy savings?

    Plots produced:
      power_per_episode.svg     – mean TX power and SINR per episode
      power_sinr_scatter.svg    – TX power vs SINR (FIX #7: replaces
                                  the meaningless action-vs-throughput scatter)
      power_final_episode.svg   – TX power and SINR within the final episode
    """
    summaries = [episode_summary(df) for df in dfs]

    # ── Per-episode mean TX power and SINR ────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for s, df, lab, color in zip(summaries, dfs, labels, PALETTE):
        lab = _label(lab, df)
        axes[0].plot(s["episode"], s["mean_txpower"],
                     color=color, linewidth=1.5, label=lab)
        axes[1].plot(s["episode"], s["mean_sinr"],
                     color=color, linewidth=1.5, label=lab)
    axes[0].set_ylabel("Mean TX power (dBm)")
    axes[1].set_ylabel("Mean SINR (dB)")
    axes[-1].set_xlabel("Episode")
    for ax in axes:
        ax.legend(fontsize=16)
    fig.suptitle("TX Power and SINR per Episode")
    _save(fig, outdir, "power_per_episode.svg")

    # ── TX power vs SINR scatter (FIX #7) ─────────────────────────────────────
    # Each point is one episode's mean.  A well-trained agent should use the
    # minimum TX power that still achieves an acceptable SINR.
    fig, ax = plt.subplots(figsize=(7, 5))
    for s, df, lab, color in zip(summaries, dfs, labels, PALETTE):
        lab   = _label(lab, df)
        n_ep  = len(s)
        # Colour-code early (light) vs late (dark) episodes
        alphas = np.linspace(0.2, 1.0, n_ep)
        for i, (_, row) in enumerate(s.iterrows()):
            ax.scatter(row["mean_txpower"], row["mean_sinr"],
                       color=color, alpha=alphas[i], s=20)
        # Legend proxy
        ax.scatter([], [], color=color, label=lab, s=30)
    ax.set_xlabel("Mean TX power (dBm)")
    ax.set_ylabel("Mean SINR (dB)")
    ax.set_title("TX Power vs SINR  (light = early, dark = late episodes)\n"
                 "Goal: bottom-right cluster = low power, good SINR")
    ax.legend()
    _save(fig, outdir, "power_sinr_scatter.svg")

    # ── TX power and SINR within the final episode (single-file mode) ─────────
    if len(dfs) == 1:
        df  = derive(dfs[0])
        ep0 = df[df["episode"] == df["episode"].max()]

        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax2 = ax1.twinx()
        ax1.plot(ep0["sim_time_s"], ep0["txPowerDbm"],
                 color=PALETTE[0], linewidth=1.5, label="TX power (dBm)")
        ax2.plot(ep0["sim_time_s"], ep0["sinr"],
                 color=PALETTE[1], linewidth=1.5, linestyle="--", label="SINR (dB)")
        if var_traffic:
            _phase_lines_time(ax1)
        ax1.set_xlabel("Simulation time (s)")
        ax1.set_ylabel("TX power (dBm)", color=PALETTE[0])
        ax2.set_ylabel("SINR (dB)",      color=PALETTE[1])
        lines  = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], fontsize=16)
        title  = f"{labels[0]} – TX power and SINR in final episode"
        if var_traffic:
            title += "\n(grey dashed = traffic phase boundary)"
        fig.suptitle(title)
        _save(fig, outdir, "power_final_episode.svg")


# ── Plot group 4: Energy ───────────────────────────────────────────────────────

def plot_energy(dfs, labels, outdir):
    """
    Energy consumption and efficiency per episode.
    The Pareto plot (FIX #8) shows energy vs a single QoS score rather
    than three separate scatter plots.

    Plots produced:
      energy_per_episode.svg   – mean step energy per episode
      energy_efficiency.svg    – Mbps/J per episode
      energy_qos_pareto.svg    – FIX #8: energy vs QoS tradeoff
    """
    summaries = [episode_summary(df) for df in dfs]

    # ── Mean step energy per episode ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3))
    for s, df, lab, color in zip(summaries, dfs, labels, PALETTE):
        lab = _label(lab, df)
        ax.plot(s["episode"], s["mean_energy"],
                color=color, linewidth=1.5, label=lab)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean step energy (J)")
    ax.set_title("Energy Consumption per Episode  (lower = better)")
    ax.legend()
    _save(fig, outdir, "energy_per_episode.svg")

    # ── Throughput efficiency per episode ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3))
    for s, df, lab, color in zip(summaries, dfs, labels, PALETTE):
        lab  = _label(lab, df)
        eff  = (s["mean_thr"] / 1e6) / s["mean_energy"].replace(0, np.nan)
        ax.plot(s["episode"], eff, color=color, linewidth=1.5, label=lab)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Efficiency (Mbps / J)")
    ax.set_title("Throughput Efficiency per Episode  (higher = better)")
    ax.legend()
    _save(fig, outdir, "energy_efficiency.svg")

    # ── FIX #8: Energy vs QoS Pareto ─────────────────────────────────────────
    # One scatter per episode; each axis is a single meaningful scalar.
    # X = energy cost, Y = QoS score (higher = better QoS).
    # Ideal agent: bottom-right cluster (low energy, high QoS).
    fig, ax = plt.subplots(figsize=(8, 5))
    for df, lab, color in zip(dfs, labels, PALETTE):
        lab    = _label(lab, df)
        d      = derive(df)
        s      = d.groupby("episode", as_index=False).agg(
            mean_energy = ("stepEnergyJ", "mean"),
            mean_qos    = ("qos_score",   "mean"),
        )
        n_ep   = len(s)
        alphas = np.linspace(0.2, 1.0, n_ep)
        for i, (_, row) in enumerate(s.iterrows()):
            ax.scatter(row["mean_energy"], row["mean_qos"],
                       color=color, alpha=alphas[i], s=25)
        ax.scatter([], [], color=color, label=lab, s=40)
    ax.set_xlabel("Mean step energy (J)")
    ax.set_ylabel("Mean QoS score (higher = better)")
    ax.set_title("Energy vs QoS Tradeoff  (light = early, dark = late episodes)\n"
                 "Goal: top-left cluster = low energy, good QoS")
    ax.legend()
    _save(fig, outdir, "energy_qos_pareto.svg")


# ── Plot group 5: Variable-traffic phase analysis ──────────────────────────────

def plot_var_traffic(dfs, labels, outdir):
    """
    FIX #9: Phase-specific analysis for the VarTraffic scenarios.
    Segments the final episode into three phases and compares agent
    behaviour within each phase.

    Plots produced:
      vt_phase_comparison.svg  – box plots of reward, delay, TX power
                                 broken out by traffic phase
      vt_adaptation.svg        – rolling metrics across the episode with
                                 phase boundaries annotated
    """
    phase_edges = [0.0] + PHASE_BOUNDARIES_S + [9999.0]

    def phase_label(t):
        if t < 20.0: return "Phase 1\n(light, 512 B)"
        if t < 40.0: return "Phase 2\n(heavy, 1500 B)"
        return             "Phase 3\n(medium/bursty)"

    # ── Per-phase box plots ───────────────────────────────────────────────────
    metrics = [
        ("rew",        "Step reward"),
        ("delay_ms",   "Delay (ms)"),
        ("txPowerDbm", "TX power (dBm)"),
        ("sinr",       "SINR (dB)"),
    ]
    phase_names = [
        "Phase 1\n(light, 512 B)",
        "Phase 2\n(heavy, 1500 B)",
        "Phase 3\n(medium/bursty)",
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 10), sharex=True)
    for (df, lab), color in zip(zip(dfs, labels), PALETTE):
        d = derive(df)
        # Use last 10 episodes only (most trained behaviour)
        last_eps = d["episode"].max()
        d = d[d["episode"] > max(0, last_eps - 10)].copy()
        d["phase"] = pd.cut(
            d["sim_time_s"],
            bins=phase_edges,
            labels=phase_names,
            right=False,
        )
        for ax, (col, ylabel) in zip(axes, metrics):
            phase_data = [
                d[d["phase"] == p][col].dropna().values
                for p in phase_names
            ]
            bp = ax.boxplot(
                phase_data,
                positions=[0, 1, 2],
                widths=0.35,
                patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.5),
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
                flierprops=dict(marker=".", color=color, alpha=0.3, markersize=3),
            )
            ax.set_ylabel(ylabel)

    axes[-1].set_xticks([0, 1, 2])
    axes[-1].set_xticklabels(phase_names)
    axes[-1].set_xlabel("Traffic phase")
    fig.suptitle("Agent Behaviour by Traffic Phase\n"
                 "(last 10 episodes; box = IQR, line = median)")
    _save(fig, outdir, "vt_phase_comparison.svg")

    # ── Adaptation trace across one episode ──────────────────────────────────
    if len(dfs) == 1:
        d   = derive(dfs[0])
        ep0 = d[d["episode"] == d["episode"].max()]

        trace_metrics = [
            ("rew",        "Reward"),
            ("txPowerDbm", "TX power (dBm)"),
            ("sinr",       "SINR (dB)"),
            ("thr_mbps",   "Throughput (Mbps)"),
            ("delay_ms",   "Delay (ms)"),
        ]
        fig, axes = plt.subplots(len(trace_metrics), 1,
                                 figsize=(11, 10), sharex=True)
        for ax, (col, ylabel) in zip(axes, trace_metrics):
            ax.plot(ep0["sim_time_s"], _roll(ep0[col], 5),
                    color=PALETTE[0], linewidth=1.5)
            _phase_lines_time(ax)
            ax.set_ylabel(ylabel)

        # Annotate phases on the top axis
        for t, name in zip(
                [5.0, 25.0, 45.0],
                ["Phase 1\nlight", "Phase 2\nheavy", "Phase 3\nbursty"]
        ):
            axes[0].text(t, axes[0].get_ylim()[1], name,
                         fontsize=16, va="top", ha="center", color="grey")

        axes[-1].set_xlabel("Simulation time (s)")
        fig.suptitle(f"{labels[0]} – Adaptation across traffic phases (final episode)")
        _save(fig, outdir, "vt_adaptation.svg")


# ── Plot group 6: DQN-specific ─────────────────────────────────────────────────

def plot_dqn_specific(dfs, labels, outdir):
    """
    Epsilon decay and Q-loss.  Skipped gracefully for PPO files.

    Plots produced:
      dqn_epsilon.svg          – epsilon vs global step
      dqn_q_loss.svg           – smoothed Q-loss vs global step
      dqn_action_dist.svg      – FIX #2: action index distribution as
                                 normalised histogram per agent (no colour
                                 index arithmetic)
    """
    dqn_dfs    = [(df, lab) for df, lab in zip(dfs, labels) if _has(df, "epsilon")]
    dqn_colors = [PALETTE[i] for i, (df, _) in enumerate(zip(dfs, labels))
                  if _has(df, "epsilon")]

    if not dqn_dfs:
        return

    # ── Epsilon decay ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3))
    for (df, lab), color in zip(dqn_dfs, dqn_colors):
        ax.plot(df["gstep"],
                pd.to_numeric(df["epsilon"], errors="coerce"),
                color=color, linewidth=1.5, label=f"{lab} (DQN)")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Epsilon")
    ax.set_title("DQN Exploration: Epsilon Decay")
    ax.legend()
    _save(fig, outdir, "dqn_epsilon.svg")

    # ── Q-loss ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3))
    for (df, lab), color in zip(dqn_dfs, dqn_colors):
        ql = pd.to_numeric(df["q_loss"], errors="coerce")
        if ql.notna().any():
            ax.plot(df["gstep"], _roll(ql.fillna(method="ffill"), 50),
                    color=color, linewidth=1.5, label=f"{lab} (DQN)")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Smooth L1 loss")
    ax.set_title("DQN Q-Loss  (should trend downward and stabilise)")
    ax.legend()
    _save(fig, outdir, "dqn_q_loss.svg")

    # ── FIX #2: action index distribution ─────────────────────────────────────
    # Use a simple normalised histogram per agent; no colour index arithmetic.
    fig, ax = plt.subplots(figsize=(9, 4))
    n_agents = len(dqn_dfs)
    bar_w    = 0.8 / max(n_agents, 1)
    for i, ((df, lab), color) in enumerate(zip(dqn_dfs, dqn_colors)):
        counts = df["act_idx"].value_counts().sort_index()
        counts = counts / counts.sum()
        offset = (i - (n_agents - 1) / 2) * bar_w
        ax.bar(counts.index + offset, counts.values,
               width=bar_w, alpha=0.75, color=color, label=f"{lab} (DQN)")
    ax.set_xlabel("Action index  →  m = 0.0 (index 0) … 2.0 (index 20)")
    ax.set_ylabel("Fraction of steps")
    ax.set_title("DQN Discrete Action Distribution\n"
                 "Trained agent should prefer a narrow band of low-power actions")
    ax.legend()
    _save(fig, outdir, "dqn_action_dist.svg")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Plot PPO / DQN training logs for Simu5G-Gym.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--files",        nargs="+", required=True)
    ap.add_argument("--labels",       nargs="*")
    ap.add_argument("--outdir",       default="plots")
    ap.add_argument("--warmup_drop",  type=int, default=0,
                    help="Skip first N global steps from timeseries plots")
    ap.add_argument("--var_traffic",  action="store_true",
                    help="Mark phase boundaries at 20 s and 40 s "
                         "(for Train-Simple / Train-Multi / Train-Mobile)")
    args = ap.parse_args()

    # Labels
    if args.labels and len(args.labels) == len(args.files):
        labels = args.labels
    else:
        labels = [os.path.splitext(os.path.basename(f))[0] for f in args.files]

    os.makedirs(args.outdir, exist_ok=True)

    # Load
    dfs = []
    for f in args.files:
        df = load(f)
        if args.warmup_drop > 0:
            df = df[df["gstep"] >= args.warmup_drop].copy()
        dfs.append(df)

    for path, df, lab in zip(args.files, dfs, labels):
        print(f"  {path!r}  →  {len(df):,} rows, "
              f"{df['episode'].nunique()} episodes "
              f"[{_agent_type(df)}] as '{lab}'")

    # Render all plot groups
    print("\nGenerating plots...")

    plot_learning_curves(dfs, labels, args.outdir)
    plot_qos(dfs, labels, args.outdir,
             warmup_drop=args.warmup_drop,
             var_traffic=args.var_traffic)
    plot_power_control(dfs, labels, args.outdir,
                       var_traffic=args.var_traffic)
    plot_energy(dfs, labels, args.outdir)
    plot_dqn_specific(dfs, labels, args.outdir)

    if args.var_traffic:
        plot_var_traffic(dfs, labels, args.outdir)

    # Summary
    print("\n── Output files ──────────────────────────────────────────────────")
    print("  Learning curves:")
    print(f"    {args.outdir}/lc_episode_return.svg")
    print(f"    {args.outdir}/lc_reward_trend.svg")
    print(f"    {args.outdir}/lc_reward_distribution.svg")
    print("  QoS:")
    print(f"    {args.outdir}/qos_per_episode.svg")
    if len(dfs) == 1:
        print(f"    {args.outdir}/qos_final_episode.svg")
    print("  Power control:")
    print(f"    {args.outdir}/power_per_episode.svg")
    print(f"    {args.outdir}/power_sinr_scatter.svg")
    if len(dfs) == 1:
        print(f"    {args.outdir}/power_final_episode.svg")
    print("  Energy:")
    print(f"    {args.outdir}/energy_per_episode.svg")
    print(f"    {args.outdir}/energy_efficiency.svg")
    print(f"    {args.outdir}/energy_qos_pareto.svg")
    print("  DQN-specific (if applicable):")
    print(f"    {args.outdir}/dqn_epsilon.svg")
    print(f"    {args.outdir}/dqn_q_loss.svg")
    print(f"    {args.outdir}/dqn_action_dist.svg")
    if args.var_traffic:
        print("  Variable-traffic phase analysis:")
        print(f"    {args.outdir}/vt_phase_comparison.svg")
        if len(dfs) == 1:
            print(f"    {args.outdir}/vt_adaptation.svg")
    print(f"\nAll plots written to: {args.outdir}/")


if __name__ == "__main__":
    main()
