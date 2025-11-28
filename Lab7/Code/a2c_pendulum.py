#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 1: A2C
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh


import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.distributions import Categorical
import argparse
import wandb
from tqdm import tqdm
from typing import Tuple

def mish(input):
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, input): return mish(input)
    
def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialize."""
        super(Actor, self).__init__()
        ############TODO#############
        # 建立神經網絡結構 (連續動作空間的標準結構)
        self.model = nn.Sequential(
            nn.Linear(in_dim, 64),
            # nn.Tanh(),
            Mish(),
            nn.Linear(64, 64),
            # nn.Tanh(),
            Mish(),
            nn.Linear(64, out_dim)
        )
        # 初始化標準差參數 (用於連續動作空間的參數化高斯策略)
        self.log_std = nn.Parameter(torch.zeros(out_dim))
        
        # 初始化權重
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                initialize_uniformly(layer)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        ############TODO#############
        # 計算動作均值
        mean = self.model(state)
        # 計算標準差 (由可學習參數 log_std 轉換)
        std = torch.exp(self.log_std)
        # 創建高斯分佈
        dist = Normal(mean, std)
        # dist = Categorical(mean, std)
        # 從分佈中採樣動作
        action = dist.sample()
        #############################
        return action, dist



class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()
        ############TODO#############
        # 建立價值網絡
        self.model = nn.Sequential(
            nn.Linear(in_dim, 64),
            # nn.Tanh(),
            Mish(),
            nn.Linear(64, 64),
            # nn.Tanh(),
            Mish(),
            nn.Linear(64, 1)
        )
        
        # 初始化權重
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                initialize_uniformly(layer)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        ############TODO#############
        # 返回狀態的估計價值
        value = self.model(state)
        #############################
        return value
    

class A2CAgent:
    """A2CAgent interacting with environment.

    Atribute:
        env (gym.Env): openAI Gym environment
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        device (torch.device): cpu / gpu
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args=None):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.num_episodes = args.num_episodes
        self.max_grad_norm = args.max_grad_norm
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # transition (state, log_prob, next_state, reward, done)
        self.transition: list = list()

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            log_prob = dist.log_prob(selected_action).sum(dim=-1)
            self.transition = [state, log_prob]

        return selected_action.clamp(-2.0, 2.0).cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition.extend([next_state, reward, done])

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the model by gradient descent."""
        state, log_prob, next_state, reward, done = self.transition
        # 將 numpy 數組轉換為 torch 張量 (state 和 log_prob 已經是張量)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)      
        # Q_t = r + gamma * V(s_{t+1}) if state != Terminal
        # = r otherwise
        mask = 1 - done     
        ############TODO#############
        # value_loss = ?
        # 計算當前狀態的價值
        current_value = self.critic(state)
        # 計算下一個狀態的價值
        next_value = self.critic(next_state)
        # 計算 TD 目標
        target = reward + self.gamma * next_value * mask       
        # 計算價值損失
        value_loss = F.mse_loss(current_value, target.detach())
        #############################  
        # update value
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        # advantage = Q_t - V(s_t)
        ############TODO#############
        # 計算優勢函數 (advantage)
        advantage = (target - current_value).detach()        
        # 計算策略損失
        _, dist = self.actor(state)
        entropy = dist.entropy().mean()
        policy_loss = -(log_prob * advantage) - self.entropy_weight * entropy
        #############################
        
        # update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        return policy_loss.item(), value_loss.item()


    def train(self):
        """Train the agent."""
        self.is_test = False
        step_count = 0
        
        recent_scores = []  # 新增：用來追蹤最近 20 個 episode 的分數
        model_saved = False  # 新增：避免重複儲存模型
        
        for ep in tqdm(range(1, self.num_episodes)): 
            actor_losses, critic_losses, scores = [], [], []
            state, _ = self.env.reset(seed=self.seed)
            score = 0
            done = False
            while not done:
                self.env.render()  # Render the environment
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

                state = next_state
                score += reward
                step_count += 1
                # W&B logging
                wandb.log({
                    "step": step_count,
                    "actor loss": actor_loss,
                    "critic loss": critic_loss,
                    }) 
                # if episode ends
                if done:
                    scores.append(score)
                    print(f"Episode {ep}: Total Reward = {score}")
                    # W&B logging
                    wandb.log({
                        "episode": ep,
                        "return": score
                        }) 
            recent_scores.append(score)
            if len(recent_scores) > 20:
                recent_scores.pop(0)
            if len(recent_scores) == 20 and all(r > -150 for r in recent_scores):
                model_name = f"LAB7_task1_a2c_ep{ep}_step{step_count}.pt"  # 儲存加上 episode 編號
                print(f"\n✅ Saving model at episode {ep}: {model_name}")
                torch.save(self.actor.state_dict(), model_name)

    def test(self, video_folder: str, model_path: str):
        """Test the agent."""
        self.is_test = True
        self.actor.load_state_dict(torch.load(model_path))
        tmp_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

        self.env = tmp_env

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-a2c-run")
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=float, default=1e-2) # entropy can be disabled by setting this to 0
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--model-path", type=str, default=None, help="Path to saved model to test")  # 若有輸入 pt 檔，就執行 test
    args = parser.parse_args()
    
    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = 77
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    wandb.init(project="DLP-Lab7-A2C-Pendulum", name=args.wandb_run_name, save_code=True)
    
    agent = A2CAgent(env, args)
    if args.model_path:
        agent.test(video_folder="test_videos", model_path=args.model_path)  # 有指定 pt 檔案就測試
    else:
        agent.train()  # 沒有就訓練