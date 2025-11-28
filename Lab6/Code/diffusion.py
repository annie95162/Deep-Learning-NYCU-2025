import torch

class DiffusionSchedule:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def get_params(self, t):
        return self.betas[t], self.alphas[t], self.alpha_bars[t]

def q_sample(x_start, t, noise, alpha_bars):
    # x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * noise
    sqrt_ab = torch.sqrt(alpha_bars[t]).reshape(-1, 1, 1, 1)
    sqrt_1_ab = torch.sqrt(1 - alpha_bars[t]).reshape(-1, 1, 1, 1)
    return sqrt_ab * x_start + sqrt_1_ab * noise
