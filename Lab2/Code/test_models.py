import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
import os
import numpy as np
from utils import show_prediction

def load_best_model(model, model_path, device):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return model

def dice_score(preds, targets, threshold=0.5, eps=1e-7):
    # 確保 preds 是 Tensor 類型
    if isinstance(preds, np.ndarray):
        preds = torch.tensor(preds)

    preds = torch.sigmoid(preds)  # Apply sigmoid to get probabilities
    preds = (preds > threshold).float()  # Convert to binary mask

    # 確保 targets 是 Tensor 類型
    if isinstance(targets, np.ndarray):
        targets = torch.tensor(targets)

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def visualize_predictions(test_loader, model, device, image_index=7):
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if idx == image_index:  # 控制顯示哪一張圖片
                images = batch["image"].to(device, dtype=torch.float)
                masks = batch["mask"].to(device, dtype=torch.float)

                # Make predictions
                preds = model(images)
                preds = torch.sigmoid(preds)  # Apply sigmoid
                preds = (preds > 0.5).float()  # Convert to binary mask

                images = images.cpu().numpy()
                preds = preds.cpu().numpy()
                masks = masks.cpu().numpy()

                dice = dice_score(preds, masks)
                '''
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                show_prediction(images[0], preds[0], title=f"Sample {idx} - Predicted")  # Show image and prediction
                plt.subplot(1, 3, 2)
                show_prediction(images[0], masks[0], title="Ground Truth Mask")
                '''
                cmap = plt.cm.colors.ListedColormap(['purple', 'yellow'])

                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(images[0].transpose(1, 2, 0))  # Convert CHW to HWC
                plt.title(f'{args.model} - test({idx})')
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(preds[0][0], cmap=cmap)  # Predicted mask
                plt.title(f'Predicted Mask \n Dice Score: {dice:.4f}')
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(masks[0][0], cmap=cmap)  # Ground truth mask
                plt.title('Ground Truth Mask')
                plt.axis('off')
                
                plt.show()
                break  # Show only the selected image

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the dataset
    test_dataset = load_dataset('../dataset', mode="test")  # fixed dataset path
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if args.model == "unet":
        model = UNet(in_channels=3, out_channels=1)
    elif args.model == "resnet":
        model = ResNet34_UNet(in_channels=3, out_channels=1)
    else:
        raise ValueError("Invalid model type. Choose either 'unet' or 'resnet'.")

    model.to(device)

    # Load the best model
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    model_path = os.path.join(parent_dir, 'saved_models', f'best_{args.model}.pth')
    model = load_best_model(model, model_path, device)

    visualize_predictions(test_loader, model, device, image_index=args.image_index)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test segmentation model and visualize results')
    parser.add_argument('--model', type=str, choices=['unet', 'resnet'], required=True, help='Model type: unet or resnet')
    parser.add_argument('--image_index', type=int, default=7, help='Index of the image to display (default: 7)')
    args = parser.parse_args()
    main(args)
