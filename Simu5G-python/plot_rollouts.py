#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CANON_COLS = [
    "thr", "delay", "jitter", "loss", "numUe", "stepEnergyJ", "rew",
    "act", "act_m", "txPowerDbm", "sinr"
]


def load_rollout(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # normalize timestamp column name
    if "ts" not in df.columns and "t" in df.columns:
        df = df.rename(columns={"t": "ts"})

    # ensure episode/step exist
    if "episode" not in df.columns:
        df["episode"] = 0
    if "step" not in df.columns:
        df["step"] = np.arange(len(df), dtype=int)

    # keep only known columns if present (don’t crash if extra cols exist)
    # also ensure required metrics exist
    for c in CANON_COLS:
        if c not in df.columns:
            df[c] = np.nan
    if df["act"].isna().all() and "act_m" in df.columns:
        df["act"] = df["act_m"]

    # sort and build a global step axis based on (episode, step)
    df = df.sort_values(["episode", "step"]).reset_index(drop=True)

    # infer horizon (max step + 1) across episodes (works for fixed length episodes like 200)
    horizon = int(df.groupby("episode")["step"].max().max() + 1)
    df["gstep"] = df["episode"] * horizon + df["step"]

    return df


def derive_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["thr_mbps"] = out["thr"] / 1e6
    out["delay_ms"] = out["delay"] * 1e3
    out["jitter_ms"] = out["jitter"] * 1e3
    # Offered load proxy: if loss is a delivery ratio, offered ~= received / (1 - loss)
    denom = (1.0 - out["loss"]).clip(lower=1e-6)
    out["offered_mbps"] = out["thr_mbps"] / denom
    out["delivery_ratio"] = (1.0 - out["loss"]).clip(lower=0.0, upper=1.0)
    energy = out["stepEnergyJ"].replace(0, np.nan)
    out["eff_mbps_per_j"] = out["thr_mbps"] / energy
    return out


def per_episode_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("episode", as_index=False)
    out = g.agg(
        steps=("step", "max"),
        ep_return=("rew", "sum"),
        mean_thr=("thr", "mean"),
        mean_delay=("delay", "mean"),
        mean_jitter=("jitter", "mean"),
        mean_loss=("loss", "mean"),
        mean_energy=("stepEnergyJ", "mean"),
    )
    out["steps"] = out["steps"] + 1  # max step -> count
    return out


def plot_timeseries(dfs, labels, outdir, warmup_drop=0):
    # Continuous traces across episodes using gstep
    metrics = [
        ("thr", "Throughput"),
        ("delay", "Frame delay (s)"),
        ("stepEnergyJ", "Step energy (J)"),
        ("rew", "Reward"),
    ]

    for col, title in metrics:
        plt.figure()
        for df, lab in zip(dfs, labels):
            d = df
            if warmup_drop > 0:
                d = d[d["gstep"] >= warmup_drop]
            if d[col].isna().all():
                continue
            plt.plot(d["gstep"], d[col], label=lab)
        plt.xlabel("Global step (episode*horizon + step)")
        plt.ylabel(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"timeseries_{col}.png"), dpi=150)
        plt.close()

    # Cumulative reward over time
    plt.figure()
    for df, lab in zip(dfs, labels):
        d = df
        if warmup_drop > 0:
            d = d[d["gstep"] >= warmup_drop]
        if d["rew"].isna().all():
            continue
        plt.plot(d["gstep"], d["rew"].cumsum(), label=lab)
    plt.xlabel("Global step (episode*horizon + step)")
    plt.ylabel("Cumulative reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "timeseries_rew_cum.png"), dpi=150)
    plt.close()


def plot_episode_curves(dfs, labels, outdir):
    # Per-episode return + per-episode mean metrics with error bars
    summaries = [per_episode_summary(df) for df in dfs]

    # Episode return with rolling average (TRAINING PROGRESS INDICATOR)
    plt.figure(figsize=(10, 5))
    for s, lab in zip(summaries, labels):
        # Plot raw episode returns
        plt.plot(s["episode"], s["ep_return"], marker="o", label=f"{lab} (raw)", alpha=0.5, linewidth=1)
        
        # Plot rolling average (window=10)
        if len(s) >= 10:
            rolling_mean = s["ep_return"].rolling(window=10, center=True).mean()
            plt.plot(s["episode"], rolling_mean, linewidth=2.5, label=f"{lab} (rolling avg)")
    
    plt.xlabel("Episode")
    plt.ylabel("Episode return (sum of reward)")
    plt.title("Episode Return - Shows Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "per_episode_return_with_trend.png"), dpi=150)
    plt.close()
    
    # Episode return trend (simple learning curve)
    plt.figure(figsize=(10, 5))
    for s, lab in zip(summaries, labels):
        plt.plot(s["episode"], s["ep_return"], marker="o", label=lab, linewidth=2)
    
    plt.xlabel("Episode")
    plt.ylabel("Episode return (sum of reward)")
    plt.title("Learning Curve - Is the Model Training?")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "learning_curve_episode_return.png"), dpi=150)
    plt.close()

    # Mean throughput per episode
    plt.figure()
    for s, lab in zip(summaries, labels):
        plt.plot(s["episode"], s["mean_thr"], marker="o", label=lab)
    plt.xlabel("Episode")
    plt.ylabel("Mean throughput")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "per_episode_mean_thr.png"), dpi=150)
    plt.close()

    # Mean delay per episode
    plt.figure()
    for s, lab in zip(summaries, labels):
        plt.plot(s["episode"], s["mean_delay"], marker="o", label=lab)
    plt.xlabel("Episode")
    plt.ylabel("Mean delay (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "per_episode_mean_delay.png"), dpi=150)
    plt.close()

    # Mean energy per episode
    plt.figure()
    for s, lab in zip(summaries, labels):
        plt.plot(s["episode"], s["mean_energy"], marker="o", label=lab)
    plt.xlabel("Episode")
    plt.ylabel("Mean step energy (J)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "per_episode_mean_energy.png"), dpi=150)
    plt.close()


def plot_training_metrics(dfs, labels, outdir):
    """Plot key metrics for understanding if the model is training."""
    
    # 1. Reward per step over time (all steps, not per-episode)
    plt.figure(figsize=(12, 5))
    for df, lab in zip(dfs, labels):
        plt.scatter(df["gstep"], df["rew"], s=5, alpha=0.4, label=lab)
    plt.xlabel("Global step")
    plt.ylabel("Reward per step")
    plt.title("Reward per Step - Higher values = Model is learning")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "reward_per_step.png"), dpi=150)
    plt.close()
    
    # 2. Moving average of reward (100-step window)
    plt.figure(figsize=(12, 5))
    for df, lab in zip(dfs, labels):
        if len(df) > 0:
            moving_avg = df["rew"].rolling(window=100, min_periods=1).mean()
            plt.plot(df["gstep"], moving_avg, linewidth=2, label=f"{lab} (100-step avg)")
    plt.xlabel("Global step")
    plt.ylabel("Average reward")
    plt.title("Moving Average Reward (100 steps) - Clear Training Signal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "reward_moving_average.png"), dpi=150)
    plt.close()
    
    # 3. Cumulative steps per episode (episode progression)
    plt.figure(figsize=(10, 5))
    for df, lab in zip(dfs, labels):
        df_sorted = df.sort_values(["episode", "step"])
        steps_per_ep = df_sorted.groupby("episode").size()
        cumsteps = steps_per_ep.cumsum()
        plt.plot(cumsteps.index, cumsteps.values, marker="o", label=lab)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative steps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "episode_progression.png"), dpi=150)
    plt.close()
    
    # 4. Mean reward per episode (alternative view)
    plt.figure(figsize=(10, 5))
    for df, lab in zip(dfs, labels):
        mean_rew_per_ep = df.groupby("episode")["rew"].mean()
        plt.plot(mean_rew_per_ep.index, mean_rew_per_ep.values, marker="o", linewidth=2, label=lab)
    plt.xlabel("Episode")
    plt.ylabel("Mean reward per episode")
    plt.title("Average Episode Reward - Should increase if training")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mean_reward_per_episode.png"), dpi=150)
    plt.close()
    
    # 5. Reward vs Efficiency (Energy-aware metric)
    plt.figure(figsize=(10, 5))
    for df, lab in zip(dfs, labels):
        # Group by episode and compute episode-level stats
        ep_data = []
        for ep in sorted(df["episode"].unique()):
            ep_df = df[df["episode"] == ep]
            total_rew = ep_df["rew"].sum()
            total_energy = ep_df["stepEnergyJ"].sum()
            if total_energy > 0:
                efficiency = total_rew / total_energy
                ep_data.append({"episode": ep, "reward": total_rew, "efficiency": efficiency})
        
        if ep_data:
            ep_df_temp = pd.DataFrame(ep_data)
            plt.scatter(ep_df_temp["episode"], ep_df_temp["efficiency"], s=50, alpha=0.6, label=lab)
    
    plt.xlabel("Episode")
    plt.ylabel("Reward per unit energy")
    plt.title("Energy Efficiency - Higher = Better energy-reward tradeoff")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "reward_efficiency_per_episode.png"), dpi=150)
    plt.close()
    
    # 6. Reward histogram evolution (first 10% vs last 10% of episodes)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for df, lab in zip(dfs, labels):
        n_ep = df["episode"].max() + 1
        thresh_early = int(n_ep * 0.1)
        thresh_late = int(n_ep * 0.9)
        
        early_rews = df[df["episode"] < thresh_early]["rew"]
        late_rews = df[df["episode"] >= thresh_late]["rew"]
        
        axes[0].hist(early_rews, bins=30, alpha=0.6, label=lab)
        axes[1].hist(late_rews, bins=30, alpha=0.6, label=lab)
    
    axes[0].set_title("Reward Distribution - Early Episodes")
    axes[0].set_xlabel("Reward per step")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title("Reward Distribution - Late Episodes")
    axes[1].set_xlabel("Reward per step")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle("Is the reward shifting higher? = Model learning", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "reward_distribution_evolution.png"), dpi=150)
    plt.close(fig)
    
    # 7. Comparison: cumulative reward over time
    fig, axes = plt.subplots(1, len(dfs), figsize=(6*len(dfs), 4))
    if len(dfs) == 1:
        axes = [axes]
    
    for ax, df, lab in zip(axes, dfs, labels):
        for ep in sorted(df["episode"].unique())[:min(5, df["episode"].nunique())]:
            ep_df = df[df["episode"] == ep].sort_values("step")
            ax.plot(ep_df["step"], ep_df["rew"].cumsum(), label=f"Ep {ep}", alpha=0.7)
        
        ax.set_xlabel("Step within episode")
        ax.set_ylabel("Cumulative reward")
        ax.set_title(f"{lab} - First 5 episodes")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "episode_reward_accumulation.png"), dpi=150)
    plt.close(fig)
    
    print("\n📊 TRAINING PROGRESS PLOTS CREATED:")
    print("  - reward_per_step.png: Scatter plot of all reward values")
    print("  - reward_moving_average.png: ⭐ BEST for seeing training progress (100-step rolling avg)")
    print("  - mean_reward_per_episode.png: Average reward trend per episode")
    print("  - reward_distribution_evolution.png: Compare early vs late episodes")
    print("  - episode_progression.png: How many steps completed per episode")
    print("  - reward_efficiency_per_episode.png: Reward-to-energy ratio per episode")
    print("  - episode_reward_accumulation.png: How quickly rewards accumulate within episodes")
    print()



def plot_action_distribution(dfs, labels, outdir):
    # Overall action histogram
    plt.figure()
    for df, lab in zip(dfs, labels):
        if df["act"].isna().all():
            continue
        counts = df["act"].value_counts().sort_index()
        xs = counts.index.to_numpy()
        ys = counts.values.astype(float)
        ys = ys / ys.sum() if ys.sum() > 0 else ys
        plt.plot(xs, ys, marker="o", label=lab)
    plt.xlabel("Action")
    plt.ylabel("Fraction of steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "action_distribution.png"), dpi=150)
    plt.close()

    # Action over time (PPO especially)
    plt.figure()
    for df, lab in zip(dfs, labels):
        if df["act"].isna().all():
            continue
        plt.plot(df["gstep"], df["act"], label=lab, linewidth=1)
    plt.xlabel("Global step")
    plt.ylabel("Action")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "action_over_time.png"), dpi=150)
    plt.close()


def plot_single_run(df, label, outdir, warmup_drop=0):
    d = df
    if warmup_drop > 0:
        d = d[d["gstep"] >= warmup_drop]
    d = derive_metrics(d)

    x = d["step"]

    # Throughput / Energy / Reward
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(x, d["thr_mbps"], label=label)
    axes[0].set_ylabel("Throughput (Mbps)")
    axes[1].plot(x, d["stepEnergyJ"], label=label)
    axes[1].set_ylabel("Step energy (J)")
    axes[2].plot(x, d["rew"], label=label)
    axes[2].set_ylabel("Reward")
    axes[2].set_xlabel("Step")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "single_thr_energy_reward.png"), dpi=150)
    plt.close(fig)

    # Cumulative reward
    plt.figure(figsize=(8, 3))
    plt.plot(x, d["rew"].cumsum(), label=label)
    plt.xlabel("Step")
    plt.ylabel("Cumulative reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "single_reward_cum.png"), dpi=150)
    plt.close()

    # Cumulative energy
    plt.figure(figsize=(8, 3))
    plt.plot(x, d["stepEnergyJ"].fillna(0).cumsum(), label=label)
    plt.xlabel("Step")
    plt.ylabel("Cumulative energy (J)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "single_energy_cum.png"), dpi=150)
    plt.close()

    # QoS: delay/jitter/loss
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(x, d["delay_ms"], label=label)
    axes[0].set_ylabel("Delay (ms)")
    axes[1].plot(x, d["jitter_ms"], label=label)
    axes[1].set_ylabel("Jitter (ms)")
    axes[2].plot(x, d["loss"], label=label)
    axes[2].set_ylabel("Loss")
    axes[2].set_xlabel("Step")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "single_qos.png"), dpi=150)
    plt.close(fig)

    # Action + TX power
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(x, d["act"], label="Action", color="tab:blue")
    ax.set_ylabel("Action")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    if not d["txPowerDbm"].isna().all():
        ax2 = ax.twinx()
        ax2.plot(x, d["txPowerDbm"], label="TX Power (dBm)", color="tab:orange")
        ax2.set_ylabel("TX Power (dBm)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "single_action_txpower.png"), dpi=150)
    plt.close(fig)

    # SINR
    if not d["sinr"].isna().all():
        plt.figure(figsize=(8, 3))
        plt.plot(x, d["sinr"], label=label)
        plt.xlabel("Step")
        plt.ylabel("SINR (dB)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "single_sinr.png"), dpi=150)
        plt.close()

    # Efficiency (Throughput per Joule)
    if not d["eff_mbps_per_j"].isna().all():
        plt.figure(figsize=(8, 3))
        plt.plot(x, d["eff_mbps_per_j"], label=label)
        plt.xlabel("Step")
        plt.ylabel("Mbps per J")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "single_efficiency.png"), dpi=150)
        plt.close()

    # Scatter: action vs throughput/energy
    plt.figure(figsize=(6, 4))
    plt.scatter(d["act"], d["thr_mbps"], s=10, alpha=0.6)
    plt.xlabel("Action")
    plt.ylabel("Throughput (Mbps)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "single_action_vs_thr.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(d["act"], d["stepEnergyJ"], s=10, alpha=0.6)
    plt.xlabel("Action")
    plt.ylabel("Step energy (J)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "single_action_vs_energy.png"), dpi=150)
    plt.close()

    # Scatter: action vs offered load proxy
    plt.figure(figsize=(6, 4))
    plt.scatter(d["act"], d["offered_mbps"], s=10, alpha=0.6)
    plt.xlabel("Action")
    plt.ylabel("Offered load (Mbps)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "single_action_vs_load.png"), dpi=150)
    plt.close()

    # Offered load over time
    plt.figure(figsize=(8, 3))
    plt.plot(x, d["offered_mbps"], label=label)
    plt.xlabel("Step")
    plt.ylabel("Offered load (Mbps)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "single_load_over_time.png"), dpi=150)
    plt.close()


def plot_compare_runs(dfs, labels, outdir, warmup_drop=0):
    derived = []
    for df in dfs:
        d = df
        if warmup_drop > 0:
            d = d[d["gstep"] >= warmup_drop]
        derived.append(derive_metrics(d))

    # Throughput / Energy / Reward
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for d, lab in zip(derived, labels):
        axes[0].plot(d["gstep"], d["thr_mbps"], label=lab)
        axes[1].plot(d["gstep"], d["stepEnergyJ"], label=lab)
        axes[2].plot(d["gstep"], d["rew"], label=lab)
    axes[0].set_ylabel("Throughput (Mbps)")
    axes[1].set_ylabel("Step energy (J)")
    axes[2].set_ylabel("Reward")
    axes[2].set_xlabel("Global step")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "compare_thr_energy_reward.png"), dpi=150)
    plt.close(fig)

    # QoS: delay/jitter/loss
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for d, lab in zip(derived, labels):
        axes[0].plot(d["gstep"], d["delay_ms"], label=lab)
        axes[1].plot(d["gstep"], d["jitter_ms"], label=lab)
        axes[2].plot(d["gstep"], d["loss"], label=lab)
    axes[0].set_ylabel("Delay (ms)")
    axes[1].set_ylabel("Jitter (ms)")
    axes[2].set_ylabel("Loss")
    axes[2].set_xlabel("Global step")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "compare_qos.png"), dpi=150)
    plt.close(fig)

    # Action + TX power
    fig, ax = plt.subplots(figsize=(8, 3))
    for d, lab in zip(derived, labels):
        if d["act"].isna().all():
            continue
        ax.plot(d["gstep"], d["act"], label=f"{lab} action")
    ax.set_ylabel("Action")
    ax.set_xlabel("Global step")
    ax.grid(True, alpha=0.3)
    if any(not d["txPowerDbm"].isna().all() for d in derived):
        ax2 = ax.twinx()
        for d, lab in zip(derived, labels):
            if d["txPowerDbm"].isna().all():
                continue
            ax2.plot(d["gstep"], d["txPowerDbm"], label=f"{lab} tx", linestyle="--")
        ax2.set_ylabel("TX Power (dBm)")
        lines, labs = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labs + labs2, loc="best")
    else:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "compare_action_txpower.png"), dpi=150)
    plt.close(fig)

    # SINR
    if any(not d["sinr"].isna().all() for d in derived):
        plt.figure(figsize=(8, 3))
        for d, lab in zip(derived, labels):
            if d["sinr"].isna().all():
                continue
            plt.plot(d["gstep"], d["sinr"], label=lab)
        plt.xlabel("Global step")
        plt.ylabel("SINR (dB)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "compare_sinr.png"), dpi=150)
        plt.close()

    # Efficiency (Throughput per Joule)
    if any(not d["eff_mbps_per_j"].isna().all() for d in derived):
        plt.figure(figsize=(8, 3))
        for d, lab in zip(derived, labels):
            if d["eff_mbps_per_j"].isna().all():
                continue
            plt.plot(d["gstep"], d["eff_mbps_per_j"], label=lab)
        plt.xlabel("Global step")
        plt.ylabel("Mbps per J")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "compare_efficiency.png"), dpi=150)
        plt.close()

    # Cumulative energy over time
    plt.figure(figsize=(8, 3))
    for d, lab in zip(derived, labels):
        plt.plot(d["gstep"], d["stepEnergyJ"].fillna(0).cumsum(), label=lab)
    plt.xlabel("Global step")
    plt.ylabel("Cumulative energy (J)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "compare_energy_cum.png"), dpi=150)
    plt.close()

    # Offered load over time
    plt.figure(figsize=(8, 3))
    for d, lab in zip(derived, labels):
        plt.plot(d["gstep"], d["offered_mbps"], label=lab)
    plt.xlabel("Global step")
    plt.ylabel("Offered load (Mbps)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "compare_load_over_time.png"), dpi=150)
    plt.close()

    # Scatter: action vs throughput/energy/load
    plt.figure(figsize=(6, 4))
    for d, lab in zip(derived, labels):
        plt.scatter(d["act"], d["thr_mbps"], s=10, alpha=0.5, label=lab)
    plt.xlabel("Action")
    plt.ylabel("Throughput (Mbps)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "compare_action_vs_thr.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    for d, lab in zip(derived, labels):
        plt.scatter(d["act"], d["stepEnergyJ"], s=10, alpha=0.5, label=lab)
    plt.xlabel("Action")
    plt.ylabel("Step energy (J)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "compare_action_vs_energy.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    for d, lab in zip(derived, labels):
        plt.scatter(d["act"], d["offered_mbps"], s=10, alpha=0.5, label=lab)
    plt.xlabel("Action")
    plt.ylabel("Offered load (Mbps)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "compare_action_vs_load.png"), dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True, help="CSV files (e.g. fixed_0.csv fixed_1.csv fixed_2.csv eval_ppo.csv)")
    ap.add_argument("--labels", nargs="*", help="Optional labels (same count as --files)")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--warmup_drop", type=int, default=0, help="Drop first N global steps from timeseries plots")
    args = ap.parse_args()

    labels = args.labels if args.labels and len(args.labels) == len(args.files) else [os.path.basename(f) for f in args.files]

    os.makedirs(args.outdir, exist_ok=True)

    dfs = [load_rollout(f) for f in args.files]

    single_run = len(dfs) == 1 and dfs[0]["episode"].nunique() == 1
    if single_run:
        plot_single_run(dfs[0], labels[0], args.outdir, warmup_drop=args.warmup_drop)
    else:
        plot_episode_curves(dfs, labels, args.outdir)
        plot_training_metrics(dfs, labels, args.outdir)  # ⭐ NEW: Training progress plots
        plot_compare_runs(dfs, labels, args.outdir, warmup_drop=args.warmup_drop)

    print(f"✅ Wrote plots to: {args.outdir}/")
    if not single_run:
        print("\n📈 KEY PLOTS FOR TRAINING ANALYSIS:")
        print("  ⭐ learning_curve_episode_return_with_trend.png - Shows if model is improving")
        print("  ⭐ reward_moving_average.png - Clear trend of training progress")
        print("  ⭐ reward_distribution_evolution.png - Early vs late reward patterns")
        print("  ⭐ mean_reward_per_episode.png - Episode average rewards")
    
    if single_run:
        print(" - single_thr_energy_reward.png / single_qos.png / single_action_txpower.png")
        print(" - single_reward_cum.png / single_energy_cum.png / single_sinr.png / single_efficiency.png / single_action_vs_thr.png / single_action_vs_energy.png")
        print(" - single_action_vs_load.png / single_load_over_time.png")
    else:
        print("\n📊 All plots generated:")
        print(" - compare_thr_energy_reward.png / compare_qos.png / compare_action_txpower.png")
        print(" - compare_sinr.png / compare_efficiency.png / compare_energy_cum.png / compare_load_over_time.png")
        print(" - compare_action_vs_thr.png / compare_action_vs_energy.png / compare_action_vs_load.png")



if __name__ == "__main__":
    main()
