# sample_2.py (with classifier guidance)
import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from dataset import labels_to_onehot, NUM_CLASSES
from model import ConditionalUNet
from diffusion import DiffusionSchedule

import torchvision.models as models

class GuidedEvaluator(nn.Module):
    def __init__(self, checkpoint_path='./checkpoint.pth'):
        super().__init__()
        checkpoint = torch.load(checkpoint_path)
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Sequential(
            nn.Linear(512, 24),
            nn.Sigmoid()
        )
        self.resnet18.load_state_dict(checkpoint['model'])
        self.resnet18.eval()
        for p in self.resnet18.parameters():
            p.requires_grad = False  # 不更新分類器參數

    def forward(self, images):
        # 輸入需已經是 [-1,1] 範圍
        norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return self.resnet18(norm(images))

def classifier_guidance_grad(x, y, classifier, scale=1.5):
    """
    x: (batch, 3, 64, 64), requires_grad=True
    y: (batch, 24) one-hot
    classifier: GuidedEvaluator
    scale: float, guidance strength
    回傳: 梯度 (batch, 3, 64, 64)
    """
    x = x.clone().detach().requires_grad_(True)
    out = classifier(x)
    # 多標籤情境下，對正確標籤logit求和
    log_prob = torch.log((out * y).sum(dim=1) + 1e-8)
    grad = torch.autograd.grad(log_prob.sum(), x, retain_graph=False)[0]
    return scale * grad

def sample(model, schedule, labels, device, save_dir, classifier, guidance_scale=1.5, img_names=None, show_process=False):
    model.eval()
    classifier.eval()
    n = len(labels)
    x = torch.randn(n, 3, 64, 64, device=device)
    labels_onehot = torch.stack([labels_to_onehot(l) for l in labels]).to(device)
    process_imgs = []

    for t in reversed(range(schedule.timesteps)):
        t_tensor = torch.full((n,), t, device=device, dtype=torch.long)
        x = x.detach().requires_grad_(True)
        with torch.no_grad():
            noise_pred = model(x, t_tensor, labels_onehot)
        beta = schedule.betas[t]
        alpha = schedule.alphas[t]
        alpha_bar = schedule.alpha_bars[t]
        if t > 0:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)
        # classifier guidance
        grad = classifier_guidance_grad(x, labels_onehot, classifier, scale=guidance_scale)
        # DDPM update + classifier guidance
        x = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * noise_pred + beta * grad) + torch.sqrt(beta) * z
        if show_process and t % (schedule.timesteps // 8) == 0:
            process_imgs.append(x.clone().cpu())

    # 儲存影像
    x = (x.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
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
    classifier = GuidedEvaluator().to(device)

    # 產生 test.json 和 new_test.json 的圖片
    for test_file in ['test.json', 'new_test.json']:
        with open(test_file, 'r') as f:
            labels = json.load(f)
        save_dir = f"images_guided/{test_file.split('.')[0]}"
        os.makedirs(save_dir, exist_ok=True)
        sample(model, schedule, labels, device, save_dir, classifier, guidance_scale=2.5, img_names=[str(i) for i in range(len(labels))]) #can change from 1.0~2.5

    # denoising 過程圖片
    denoise_labels = [["red sphere", "cyan cylinder", "cyan cube"]]
    save_dir = "images_guided/denoise_process"
    os.makedirs(save_dir, exist_ok=True)
    sample(model, schedule, denoise_labels, device, save_dir, classifier, guidance_scale=2.5, img_names=["denoise"], show_process=True)
