#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CANON_COLS = ["thr", "delay", "jitter", "loss", "numUe", "stepEnergyJ", "rew", "act"]


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

    # sort and build a global step axis based on (episode, step)
    df = df.sort_values(["episode", "step"]).reset_index(drop=True)

    # infer horizon (max step + 1) across episodes (works for fixed length episodes like 200)
    horizon = int(df.groupby("episode")["step"].max().max() + 1)
    df["gstep"] = df["episode"] * horizon + df["step"]

    return df


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


def plot_episode_curves(dfs, labels, outdir):
    # Per-episode return + per-episode mean metrics with error bars
    summaries = [per_episode_summary(df) for df in dfs]

    # Episode return
    plt.figure()
    for s, lab in zip(summaries, labels):
        plt.plot(s["episode"], s["ep_return"], marker="o", label=lab)
    plt.xlabel("Episode")
    plt.ylabel("Episode return (sum of reward)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "per_episode_return.png"), dpi=150)
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

    plot_timeseries(dfs, labels, args.outdir, warmup_drop=args.warmup_drop)
    plot_episode_curves(dfs, labels, args.outdir)
    plot_action_distribution(dfs, labels, args.outdir)

    print(f"Wrote plots to: {args.outdir}/")
    print(" - timeseries_thr.png / timeseries_delay.png / timeseries_stepEnergyJ.png / timeseries_rew.png")
    print(" - per_episode_return.png / per_episode_mean_thr.png / per_episode_mean_delay.png / per_episode_mean_energy.png")
    print(" - action_distribution.png / action_over_time.png")


if __name__ == "__main__":
    main()
