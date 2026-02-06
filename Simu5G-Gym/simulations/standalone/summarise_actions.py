import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILES = {
    0: "rollout_action0.csv",
    1: "rollout_action1.csv",
    2: "rollout_action2.csv",
}

def add_episode_id(df: pd.DataFrame) -> pd.DataFrame:
    # New episode marker: the "all zeros" row you get right at start
    start = (df["thr"] == 0) & (df["delay"] == 0) & (df["jitter"] == 0) & (df["loss"] == 0) & (df["stepEnergyJ"] == 0) & (df["rew"] == 0)
    # cumulative episode counter (starts at 1); subtract 1 so first episode is 0
    df = df.copy()
    df["episode"] = start.cumsum() - 1
    # drop any rows before first episode marker (if any)
    df = df[df["episode"] >= 0]
    return df

def per_episode_stats(df: pd.DataFrame) -> pd.DataFrame:
    # Drop the first “zeros” row inside each episode so it doesn’t bias means
    df2 = df[~((df["thr"] == 0) & (df["delay"] == 0) & (df["rew"] == 0) & (df["stepEnergyJ"] == 0))].copy()

    g = df2.groupby("episode", as_index=False)
    out = g.agg(
        steps=("thr", "size"),
        mean_thr=("thr", "mean"),
        mean_delay=("delay", "mean"),
        mean_jitter=("jitter", "mean"),
        mean_loss=("loss", "mean"),
        total_energyJ=("stepEnergyJ", "sum"),
        mean_reward=("rew", "mean"),
        sum_reward=("rew", "sum"),
    )
    return out

def main():
    all_action_summaries = []
    all_episode_rows = []

    for act, path in FILES.items():
        df = pd.read_csv(path)
        df = add_episode_id(df)

        ep = per_episode_stats(df)
        ep["action"] = act
        all_episode_rows.append(ep)

        # summary across episodes (your “10 runs”)
        summary = {
            "action": act,
            "episodes": ep.shape[0],
            "steps_mean": ep["steps"].mean(),
            "thr_mean": ep["mean_thr"].mean(),
            "delay_mean": ep["mean_delay"].mean(),
            "jitter_mean": ep["mean_jitter"].mean(),
            "loss_mean": ep["mean_loss"].mean(),
            "energy_total_meanJ": ep["total_energyJ"].mean(),
            "reward_mean": ep["mean_reward"].mean(),
            "return_mean": ep["sum_reward"].mean(),
        }
        all_action_summaries.append(summary)

    ep_all = pd.concat(all_episode_rows, ignore_index=True)
    summary_df = pd.DataFrame(all_action_summaries).sort_values("action")
    summary_df.to_csv("summary_by_action.csv", index=False)
    ep_all.to_csv("per_episode_stats.csv", index=False)

    print(summary_df)

    # --- Plot: Energy vs Throughput (easy PoC slide) ---
    x = summary_df["action"].values
    thr = summary_df["thr_mean"].values
    energy = summary_df["energy_total_meanJ"].values

    plt.figure()
    plt.bar(x - 0.15, thr, width=0.3)
    plt.xlabel("Action (0=ACTIVE, 1=SLEEP, 2=PAUSE)")
    plt.ylabel("Mean throughput (Bps)")
    plt.title("Mean Throughput by Fixed Action")
    plt.xticks(x)
    plt.tight_layout()
    plt.savefig("throughput_by_action.png", dpi=200)

    plt.figure()
    plt.bar(x, energy)
    plt.xlabel("Action (0=ACTIVE, 1=SLEEP, 2=PAUSE)")
    plt.ylabel("Mean total energy per episode (J)")
    plt.title("Mean Total Energy by Fixed Action")
    plt.xticks(x)
    plt.tight_layout()
    plt.savefig("energy_by_action.png", dpi=200)

    # Optional: reward / return plot too
    plt.figure()
    plt.bar(x, summary_df["return_mean"].values)
    plt.xlabel("Action (0=ACTIVE, 1=SLEEP, 2=PAUSE)")
    plt.ylabel("Mean episode return (sum of rew)")
    plt.title("Mean Episode Return by Fixed Action")
    plt.xticks(x)
    plt.tight_layout()
    plt.savefig("return_by_action.png", dpi=200)

if __name__ == "__main__":
    main()
