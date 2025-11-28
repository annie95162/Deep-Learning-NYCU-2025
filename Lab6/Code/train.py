import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset import CLEVRDataset, NUM_CLASSES
from model import ConditionalUNet
from diffusion import DiffusionSchedule, q_sample
from tqdm import tqdm

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    transform = transforms.Compose([ # [0, 1] -> [-1, 1]
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = CLEVRDataset('train.json', img_folder='train_images', transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
    model = ConditionalUNet(num_classes=NUM_CLASSES).to(device)
    schedule = DiffusionSchedule() # 建立diffusion時間表(包含 β, α 值）
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    mse = nn.MSELoss()
    epochs = 5000
    best_loss = float('inf')
    loss_list = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for img, label in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False): # tqdm顯示進度條
            img = img.to(device)
            label = label.to(device)
            t = torch.randint(0, schedule.timesteps, (img.size(0),), device=device) #  隨機生成時間步數t，決定加多少noise
            noise = torch.randn_like(img)
            noisy_img = q_sample(img, t, noise, schedule.alpha_bars.to(device)) # 用q_sample()把noise加到圖片上
            pred_noise = model(noisy_img, t, label) # 預測noisy image原本加了多少噪聲noise
            loss = mse(pred_noise, noise)
            optimizer.zero_grad() # 清空梯度
            loss.backward() # 反向傳播
            optimizer.step() # 更新參數
            running_loss += loss.item()
        avg_loss = running_loss / len(trainloader)
        loss_list.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")
        # 每50epoch存最佳模型
        if (epoch + 1) % 50 == 0:
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), f'best_model_epoch{epoch+1}.pth') #  loss是最佳就儲存
                print(f"Best model saved at epoch {epoch+1} with loss {best_loss:.4f}")

    torch.save(model.state_dict(), 'ddpm_ckpt.pth')
    # 畫loss圖
    plt.figure(figsize=(10,5))
    plt.title("Training Loss Curve")
    plt.plot(range(1, epochs+1), loss_list, label="train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss.png", dpi=300)
    plt.close()

if __name__ == '__main__':
    train()
