"""DQN from scratch on Gymnasium CarRacing (based on Mnih et al. 2013).


"""
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

ENV_ID = "CarRacing-v3" if "CarRacing-v3" in gym.registry else "CarRacing-v2"

TOTAL_STEPS = 300_000
BUFFER_SIZE = 100_000
BATCH_SIZE = 32
LR = 1e-4
GAMMA = 0.99
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 100_000
LEARNING_STARTS = 10_000
TARGET_UPDATE = 2_500
FRAME_SKIP = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(frame):
    return (frame[:84, 6:90] @ [0.299, 0.587, 0.114]).astype(np.uint8)


class Env:

    def __init__(self):
        self.env = gym.make(ENV_ID, continuous=False)
        self.frames = deque(maxlen=4)

    def reset(self):
        obs, _ = self.env.reset()
        for _ in range(50):
            obs, *_ = self.env.step(0)
        for _ in range(4):
            self.frames.append(preprocess(obs))
        return np.stack(self.frames)

    def step(self, action):
        total = 0.0
        for _ in range(FRAME_SKIP):
            obs, reward, terminated, truncated, _ = self.env.step(action)
            total += reward
            if terminated or truncated:
                break
        self.frames.append(preprocess(obs))
        return np.stack(self.frames), total, terminated, truncated


class QNetwork(nn.Module):
    def __init__(self, n_actions=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, 8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256), nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.net(x.float() / 255.0)


def main():
    env = Env()
    q_net = QNetwork().to(device)
    target_net = QNetwork().to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=LR)
    buffer = deque(maxlen=BUFFER_SIZE)
    writer = SummaryWriter("runs/DQN_CarRacing")

    state = env.reset()
    ep_reward, recent = 0.0, deque(maxlen=20)

    for step in range(1, TOTAL_STEPS + 1):
        eps = max(EPS_END, EPS_START - step / EPS_DECAY)
        if random.random() < eps:
            action = random.randrange(5)
        else:
            with torch.no_grad():
                action = q_net(torch.from_numpy(state).unsqueeze(0).to(device)).argmax().item()

        next_state, reward, terminated, truncated = env.step(action)
        buffer.append((state, action, reward, next_state, terminated))
        state = next_state
        ep_reward += reward

        if terminated or truncated:
            recent.append(ep_reward)
            writer.add_scalar("rollout/ep_rew_mean", np.mean(recent), step)
            print(f"step {step}  episode reward {ep_reward:.1f}  eps {eps:.2f}")
            state, ep_reward = env.reset(), 0.0

        if step < LEARNING_STARTS:
            continue

        batch = random.sample(buffer, BATCH_SIZE)
        s, a, r, s2, d = map(np.array, zip(*batch))
        s, s2 = torch.from_numpy(s).to(device), torch.from_numpy(s2).to(device)
        a = torch.from_numpy(a).to(device)
        r = torch.tensor(r, dtype=torch.float32, device=device)
        d = torch.tensor(d, dtype=torch.float32, device=device)

        with torch.no_grad():
            y = r + GAMMA * (1 - d) * target_net(s2).max(1).values
        q = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(q, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % TARGET_UPDATE == 0:
            target_net.load_state_dict(q_net.state_dict())
        if step % 1000 == 0:
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/epsilon", eps, step)

    torch.save(q_net.state_dict(), "dqn_car_racing.pt")
    writer.close()


if __name__ == "__main__":
    main()