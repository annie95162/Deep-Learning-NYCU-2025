# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # (N, 32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # (N, 64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # (N, 64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    
    def forward(self, x):
        return self.network(x)

class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def __len__(self):
        return len(self.buffer)
    
    def add(self, transition, error=1.0):
        ########## YOUR CODE HERE (for Task 3) ##########
        priority = (abs(error) + 1e-6) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        ########## END OF YOUR CODE (for Task 3) ##########

    def sample(self, batch_size, beta):
        ########## YOUR CODE HERE (for Task 3) ##########
        buffer_len = len(self.buffer)
        probs = self.priorities[:buffer_len]
        probs = probs / probs.sum()
        indices = np.random.choice(buffer_len, batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights
        ########## END OF YOUR CODE (for Task 3) ##########

    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ##########
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + 1e-6) ** self.alpha
        ########## END OF YOUR CODE (for Task 3) ##########
        return

class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        
        # Enhanced components for Task 3
        self.memory = PrioritizedReplayBuffer(args.memory_size, alpha=args.alpha)
        self.n_step = args.n_step
        self.n_step_buffer = deque(maxlen=self.n_step)
        
        self.q_net = DQN(self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)
        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # PER beta annealing
        self.beta_start = args.beta
        self.beta_frames = 1e6  # Anneal over 1M steps
        self.target_steps = {200000, 400000, 600000, 800000, 1000000}
        self.saved_steps = set()
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=10000):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0
            self.n_step_buffer.clear()
            
            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self.preprocessor.step(next_obs)
                
                # Store transition in n-step buffer
                self.n_step_buffer.append((state, action, reward, next_state, done))
                
                # Compute n-step return when buffer is full
                if len(self.n_step_buffer) == self.n_step:
                    n_reward, n_next_state, n_done = self.compute_n_step()
                    self.memory.add(
                        (self.n_step_buffer[0][0], self.n_step_buffer[0][1], 
                         n_reward, n_next_state, n_done),
                        error=1.0  # Initial priority
                    )
                
                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1
                if self.env_count in self.target_steps and self.env_count not in self.saved_steps:
                    self.saved_steps.add(self.env_count)
                    model_path = os.path.join(
                        self.save_dir, 
                        f"LAB5_313605019_task3_pong{self.env_count}.pt"
                    )
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"[Snapshot] Saved model at {self.env_count} steps: {model_path}")
                    wandb.log({"Env Step Count": self.env_count})

                # Train with PER
                for _ in range(self.train_per_step):
                    self.train()
                
                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
                    
                    ########## END OF YOUR CODE ##########
            
            # Handle remaining transitions in n-step buffer
            while len(self.n_step_buffer) > 0:
                n_reward, n_next_state, n_done = self.compute_n_step()
                self.memory.add(
                    (self.n_step_buffer[0][0], self.n_step_buffer[0][1], 
                     n_reward, n_next_state, n_done),
                    error=1.0
                )
                self.n_step_buffer.popleft()
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            
            ########## END OF YOUR CODE ##########
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })
            # Logging and model saving (same as original)
    def evaluate(self):  # 正確縮進在類內部
        obs, _ = self.test_env.reset()
        state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0
        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device) / 255.0
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.preprocessor.step(next_obs)
        return total_reward
    
    def compute_n_step(self):
        reward, next_state, done = 0, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        for idx in reversed(range(len(self.n_step_buffer))):
            r, n_s, d = self.n_step_buffer[idx][2:]
            reward = r + self.gamma * reward * (1 - d)
            if d:
                next_state, done = n_s, d
        return reward, next_state, done

    def train(self):
        if len(self.memory) < self.replay_start_size:
            return
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        # Beta annealing
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.env_count / self.beta_frames))
        
        # PER sampling
        transitions, indices, weights = self.memory.sample(self.batch_size, beta)
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        # Convert to tensors
        states = torch.from_numpy(np.array(states)).float().to(self.device) / 255.0
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device) / 255.0
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        ########## END OF YOUR CODE ##########
        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        #states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        #next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        #actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        #rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        #dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        #q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates
        # Double DQN target
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q_values * (1 - dones)
        
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        td_errors = current_q - target_q
        loss = (weights * td_errors.pow(2)).mean()

        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        ########## END OF YOUR CODE ##########
        # Target network update
        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.train_count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="task3_pong")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999995)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    # 新增 Task 3 專用參數
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)
    args = parser.parse_args()
    
    wandb.init(project="DLP-Lab5-DQN-Pong", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(env_name="ALE/Pong-v5", args=args)
    agent.run()
