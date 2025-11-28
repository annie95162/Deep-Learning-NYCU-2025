import matplotlib.pyplot as plt
import numpy as np

# 1. Teacher Forcing Ratio 圖
def plot_teacher_forcing_ratio(teacher_forcing_ratios, epoch_range, kl_type="None", cycle=None, ratio=None):
    title = f"Teacher Forcing Ratio ({kl_type} KL Annealing | Cycle: {cycle}, Ratio: {ratio})"
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_range, teacher_forcing_ratios, label='Teacher Forcing Ratio', color='b', marker='o')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Teacher Forcing Ratio")
    plt.legend()
    plt.grid(True)
    plt.show()

# 2. 訓練過程損失曲線
def plot_loss_curve(losses, epoch_range, kl_type="None", cycle=None, ratio=None, title="Loss Curve", label="Loss"):
    title = f"{title} ({kl_type} KL Annealing | Cycle: {cycle}, Ratio: {ratio})"
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_range, losses, label=label, color='r')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# 3. PSNR per frame 圖
def plot_psnr_per_frame(psnr_values, frame_range, kl_type="None", cycle=None, ratio=None, title="PSNR per Frame in Validation Dataset"):
    title = f"{title} ({kl_type} KL Annealing)"
    plt.figure(figsize=(10, 6))
    plt.plot(frame_range, psnr_values, label='PSNR', color='g')
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("PSNR")
    plt.legend()
    plt.grid(True)
    plt.show()
