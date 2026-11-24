"""Plot reward-over-time curves from Stable Baselines 3 TensorBoard logs.

Usage:
    python plot_tensorboard.py runs                       # CarRacing
    python plot_tensorboard.py ~/persistent/carla-gym-env/tensorboard_DQN --title "CARLA Gym SAC"
    python plot_tensorboard.py runs --tags rollout/ep_rew_mean eval/mean_reward --out plots/car_racing.png

Each subfolder with event files (e.g. PPO_CarRacing_1) is drawn as its own line.
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LABELS = {
    "rollout/ep_rew_mean": "Training: mean episode reward (last {window} episodes)",
    "eval/mean_reward": "Evaluation: mean reward (deterministic policy)",
    "train/loss": "Training loss (TD error, MSE)",
}
YLABELS = {"train/loss": "Loss"}


def find_runs(logdir):
    files = glob.glob(os.path.join(os.path.expanduser(logdir), "**", "events.out.tfevents.*"), recursive=True)
    return sorted({os.path.dirname(f) for f in files})


def smooth(values, weight):
    out, last = [], values[0] if values else 0
    for v in values:
        last = weight * last + (1 - weight) * v
        out.append(last)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logdir")
    p.add_argument("--tags", nargs="+", default=["rollout/ep_rew_mean", "eval/mean_reward"])
    p.add_argument("--smooth", type=float, default=0.6, help="0 = raw, 0.9 = very smooth")
    p.add_argument("--title", default=None)
    p.add_argument("--rew-window", type=int, default=100, help="episodes in ep_rew_mean (SB3: 100)")
    p.add_argument("--out", default=None, help="PNG path (default: plots/<logdir name>.png)")
    args = p.parse_args()

    runs = find_runs(args.logdir)
    if not runs:
        raise SystemExit(f"No TensorBoard event files under {args.logdir}")

    data = {}  # tag -> [(run name, steps, values)]
    for run in runs:
        acc = EventAccumulator(run, size_guidance={"scalars": 0})
        acc.Reload()
        available = acc.Tags()["scalars"]
        for tag in args.tags:
            if tag in available:
                events = acc.Scalars(tag)
                data.setdefault(tag, []).append(
                    (os.path.relpath(run, args.logdir), [e.step for e in events], [e.value for e in events])
                )

    if not data:
        raise SystemExit(f"None of {args.tags} found. Available tags: {available}")

    fig, axes = plt.subplots(len(data), 1, figsize=(8, 3.5 * len(data)), squeeze=False)
    for ax, (tag, series) in zip(axes[:, 0], data.items()):
        for name, steps, values in series:
            line, = ax.plot(steps, smooth(values, args.smooth), label=name, linewidth=2)
            if args.smooth > 0:
                ax.plot(steps, values, color=line.get_color(), alpha=0.25, linewidth=1)
            print(f"{tag} [{name}]: {len(values)} points, final {values[-1]:.1f}, best {max(values):.1f}")
        ax.set_title(LABELS.get(tag, tag).format(window=args.rew_window))
        ax.set_xlabel("Timesteps")
        ax.set_ylabel(YLABELS.get(tag, "Reward"))
        ax.grid(alpha=0.3)
        if len(series) > 1:
            ax.legend()

    if args.title:
        fig.suptitle(args.title, fontweight="bold")
    fig.tight_layout()

    out = args.out or os.path.join("plots", os.path.basename(os.path.normpath(args.logdir)) + ".png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
