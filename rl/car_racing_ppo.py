"""Train PPO (Stable Baselines 3) on Gymnasium CarRacing and log to TensorBoard.

Usage (inside the Nautilus desktop, `rl` conda env):
    python car_racing_ppo.py                      # train 500k steps, then evaluate
    python car_racing_ppo.py --timesteps 100000   # shorter run
    python car_racing_ppo.py --eval-only models/ppo_car_racing.zip

Reward curves go to ./runs (view with `tensorboard --logdir runs`
or plot with `python plot_tensorboard.py runs`).
"""
import argparse
import os

import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage

# Gymnasium 1.x calls it CarRacing-v3; 0.29 (Python 3.8) calls it CarRacing-v2.
ENV_ID = "CarRacing-v3" if "CarRacing-v3" in gym.registry else "CarRacing-v2"


def make_env(n_envs, seed):
    # One process per env so the CPU-bound Box2D sims run on separate cores.
    vec_cls = SubprocVecEnv if n_envs > 1 else None
    env = make_vec_env(ENV_ID, n_envs=n_envs, seed=seed, vec_env_cls=vec_cls)  # adds Monitor -> episode rewards
    env = VecFrameStack(env, n_stack=2)  # lets the policy see motion
    return VecTransposeImage(env)  # HWC -> CHW for the CNN


def linear_schedule(start):
    return lambda progress_remaining: progress_remaining * start


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logdir", default="runs")
    p.add_argument("--model-path", default="models/ppo_car_racing")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-only", metavar="MODEL_ZIP")
    args = p.parse_args()

    if args.eval_only:
        evaluate(PPO.load(args.eval_only), args)
        return

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    env = make_env(args.n_envs, args.seed)
    eval_env = make_env(1, args.seed + 1000)

    # Hyperparameters follow the RL Baselines3 Zoo settings for CarRacing.
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=linear_schedule(1e-4),
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs=dict(
            log_std_init=-2,
            ortho_init=False,
            activation_fn=nn.GELU,
            net_arch=dict(pi=[256], vf=[256]),
        ),
        tensorboard_log=args.logdir,
        seed=args.seed,
        verbose=1,
    )

    callbacks = [
        # Logs eval/mean_reward to TensorBoard and keeps the best model.
        EvalCallback(
            eval_env,
            eval_freq=max(25_000 // args.n_envs, 1),
            n_eval_episodes=5,
            best_model_save_path=os.path.dirname(args.model_path),
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=max(100_000 // args.n_envs, 1),
            save_path=os.path.dirname(args.model_path),
            name_prefix="ppo_ckpt",
        ),
    ]

    model.learn(total_timesteps=args.timesteps, callback=callbacks, tb_log_name="PPO_CarRacing")
    model.save(args.model_path)
    print(f"Saved model to {args.model_path}.zip")
    evaluate(model, args)


def evaluate(model, args):
    env = make_env(1, args.seed + 2000)
    mean, std = evaluate_policy(model, env, n_eval_episodes=args.eval_episodes, deterministic=True)
    print(f"{ENV_ID}: mean reward over {args.eval_episodes} episodes = {mean:.1f} +/- {std:.1f}")


if __name__ == "__main__":
    main()
