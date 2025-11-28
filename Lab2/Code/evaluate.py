import os
import torch
from torch.utils.data import DataLoader
from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import dice_score  # 從 utils 引入 dice 計算

def evaluate(net, data, device):
    net.eval()
    total_dice = 0.0
    count = 0

    with torch.no_grad():
        for batch in data:
            image = batch["image"].to(device, dtype=torch.float)
            mask = batch["mask"].to(device, dtype=torch.float)

            pred = net(image)
            dice = dice_score(pred, mask)
            total_dice += dice
            count += 1

    avg_dice = total_dice / count
    return avg_dice

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, choices=["unet", "resnet"], required=True)
    parser.add_argument('--mode', type=str, choices=["test", "valid"], default="test")
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if args.model_type == "unet":
        net = UNet(in_channels=3, out_channels=1)
    elif args.model_type == "resnet":
        net = ResNet34_UNet(in_channels=3, out_channels=1)
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.to(device)

    # Load dataset
    dataset = load_dataset(args.data_path, mode=args.mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Evaluate
    avg_dice = evaluate(net, dataloader, device)
    print(f"[{args.model_type.upper()}] Average Dice Score on {args.mode} set: {avg_dice:.4f}")
