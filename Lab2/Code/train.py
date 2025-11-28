import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet

def dice_score(preds, targets, threshold=0.5, eps=1e-7):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")
    # Load datasets
    train_dataset = load_dataset(args.data_path, mode="train")
    val_dataset = load_dataset(args.data_path, mode="valid")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Choose model
    if args.model == "unet":
        model = UNet(in_channels=3, out_channels=1)
    elif args.model == "resnet":
        model = ResNet34_UNet(in_channels=3, out_channels=1)
    else:
        raise ValueError("Invalid model: choose 'unet' or 'resnet'")
    
    model.to(device)

    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_dice = 0.0
    save_model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")
    os.makedirs(save_model_dir, exist_ok=True)
    # os.makedirs("saved_models", exist_ok=True)
    train_loss_history = []
    val_loss_history = []
    val_dice_history = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images = batch["image"].to(device, dtype=torch.float)
            masks = batch["mask"].to(device, dtype=torch.float)

            preds = model(images)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, dtype=torch.float)
                masks = batch["mask"].to(device, dtype=torch.float)

                preds = model(images)
                loss = criterion(preds, masks)
                val_loss += loss.item()
                val_dice += dice_score(preds, masks)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        print(f"\nEpoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Val Dice={avg_val_dice:.4f}")

        # Save best model
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            model_path = os.path.join(save_model_dir, f"best_{args.model}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model (dice={best_dice:.4f})")

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        val_dice_history.append(avg_val_dice)
    
    plot_training_history(train_loss_history, val_loss_history, val_dice_history)
    
    # Test after training with best model
    print("\n Evaluating on test set with best model...")
    # model.load_state_dict(torch.load(f"saved_models/best_{args.model}.pth", map_location=device))
    # model.load_state_dict(torch.load(f"saved_models/best_{args.model}.pth", map_location=device, weights_only=True))
    model.load_state_dict(torch.load(os.path.join(save_model_dir, f"best_{args.model}.pth"), map_location=device, weights_only=True))


    test_dataset = load_dataset(args.data_path, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()
    test_dice = 0.0
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device, dtype=torch.float)
            masks = batch["mask"].to(device, dtype=torch.float)

            preds = model(images)
            test_dice += dice_score(preds, masks)
    test_dice /= len(test_loader)
    print(f" Test Dice Score = {test_dice:.4f}")


def get_args():
    parser = argparse.ArgumentParser(description='Train a segmentation model')
    default_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    parser.add_argument('--data_path', type=str, default=default_data_path, help='path of the input data')
    parser.add_argument('--model', type=str, choices=['unet', 'resnet'], required=True, help='Model type: unet or resnet')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4, help='Learning rate')
    return parser.parse_args()

def plot_training_history(train_loss_history, val_loss_history, val_dice_history):
    epochs = range(1, len(train_loss_history) + 1)

    # Plot Train Loss
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss_history, label='Train Loss')
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Validation Loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_loss_history, label='Validation Loss', color='orange')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Validation Dice Score
    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_dice_history, label='Validation Dice', color='green')
    plt.title('Validation Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = get_args()
    train(args)

