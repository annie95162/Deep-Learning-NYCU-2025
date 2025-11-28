import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import imageio
import os
import argparse

class DQN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.network(x)

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    model = DQN(input_dim=4, num_actions=env.action_space.n).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0
        frames = []

        while not done:
            frame = env.render()
            frames.append(frame)

            obs_tensor = torch.from_numpy(np.array(obs)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(obs_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            obs = next_obs

        out_path = os.path.join(args.output_dir, f"cartpole_ep{ep}_r{int(total_reward)}.mp4")
        # with imageio.get_writer(out_path, fps=30) as video:
        with imageio.get_writer(out_path, fps=30, macro_block_size=1) as video:
            for f in frames:
                video.append_data(f)
        print(f"Saved CartPole Episode {ep} → Reward: {total_reward} → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./eval_videos_cartpole")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=313551076)
    args = parser.parse_args()
    evaluate(args)
