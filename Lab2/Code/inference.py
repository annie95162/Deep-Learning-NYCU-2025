import argparse
import torch
from torch.utils.data import DataLoader
from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import tensor_to_image, show_prediction
import os
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', required=True, help='path to the stored model weight (.pth)')
    parser.add_argument('--model_type', choices=['unet', 'resnet'], required=True, help='Model type')
    parser.add_argument('--data_path', type=str, required=True, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--mode', type=str, choices=['test', 'valid'], default='test', help='dataset split')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of images to predict')
    parser.add_argument('--save_dir', type=str, default=None, help='Folder to save masks (optional)')
    parser.add_argument('--no_show', action='store_true', help='If set, do not display images')

    return parser.parse_args()

def inference(net, dataloader, device, save_dir=None, show=True):
    net.eval()
    os.makedirs(save_dir, exist_ok=True) if save_dir else None

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            image = batch["image"].to(device, dtype=torch.float)
            orig_image = image[0].detach().cpu()
            pred = net(image)
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()

            img_np = tensor_to_image(orig_image)
            mask_np = tensor_to_image(pred[0])

            if show:
                show_prediction(orig_image, pred[0], title=f"Sample {idx}")

            if save_dir:
                plt.imsave(os.path.join(save_dir, f"sample_{idx}.png"), mask_np, cmap="Reds")

if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if args.model_type == "unet":
        model = UNet(in_channels=3, out_channels=1)
    elif args.model_type == "resnet":
        model = ResNet34_UNet(in_channels=3, out_channels=1)
    else:
        raise ValueError("Unknown model type")

    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)

    # Load dataset
    dataset = load_dataset(args.data_path, mode=args.mode)
    subset = torch.utils.data.Subset(dataset, range(args.num_samples))
    dataloader = DataLoader(subset, batch_size=1, shuffle=False)

    inference(model, dataloader, device, save_dir=args.save_dir, show=not args.no_show)
    