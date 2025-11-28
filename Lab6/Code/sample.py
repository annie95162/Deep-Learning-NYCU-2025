import os
import json
import torch
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from dataset import labels_to_onehot, NUM_CLASSES
from model import ConditionalUNet
from diffusion import DiffusionSchedule

def sample(model, schedule, labels, device, save_dir, img_names=None, show_process=False):
    model.eval()
    n = len(labels)
    x = torch.randn(n, 3, 64, 64, device=device) # 建立隨機noisy影像為初始輸入，每張影像: 3x64x64（RGB）
    labels = torch.stack([labels_to_onehot(l) for l in labels]).to(device) # one-hot encoding
    process_imgs = [] # denoise image process
    for t in reversed(range(schedule.timesteps)): # 從最雜訊回推
        t_tensor = torch.full((n,), t, device=device, dtype=torch.long)
        with torch.no_grad(): # 節省記憶體
            noise_pred = model(x, t_tensor, labels) # 預測當前影像noise
        beta = schedule.betas[t] # 對應timestep的參數
        alpha = schedule.alphas[t]
        alpha_bar = schedule.alpha_bars[t]
        if t > 0: # 不是最後一步
            z = torch.randn_like(x)
        else: # 最後一步不加雜訊
            z = torch.zeros_like(x)
        # 去雜訊公式
        x = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * noise_pred) + torch.sqrt(beta) * z
        if show_process and t % (schedule.timesteps // 8) == 0: # 至少8個 process過程
            process_imgs.append(x.clone().cpu())
    # 儲存影像
    x = (x.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1] denormalized
    for i in range(n):
        save_path = os.path.join(save_dir, f"{img_names[i] if img_names else i}.png")
        save_image(x[i], save_path)
    # 儲存grid
    grid = make_grid(x, nrow=8)
    save_image(grid, os.path.join(save_dir, "grid.png"))
    # 儲存denoising process
    if show_process and len(process_imgs) > 0:
        process_grid = make_grid(torch.cat(process_imgs, dim=0), nrow=len(process_imgs))
        process_grid = (process_grid.clamp(-1, 1) + 1) / 2
        save_image(process_grid, os.path.join(save_dir, "denoise_process.png"))
    return x

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalUNet(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load('ddpm_ckpt.pth', map_location=device))
    schedule = DiffusionSchedule()

    # 產生test.json和new_test.json的圖片
    for test_file in ['test.json', 'new_test.json']:
        with open(test_file, 'r') as f:
            labels = json.load(f)
        save_dir = f"images/{test_file.split('.')[0]}"
        os.makedirs(save_dir, exist_ok=True)
        sample(model, schedule, labels, device, save_dir, img_names=[str(i) for i in range(len(labels))])

    # denoising過程圖片
    denoise_labels = [["red sphere", "cyan cylinder", "cyan cube"]]
    save_dir = "images/denoise_process"
    os.makedirs(save_dir, exist_ok=True)
    sample(model, schedule, denoise_labels, device, save_dir, img_names=["denoise"], show_process=True)
