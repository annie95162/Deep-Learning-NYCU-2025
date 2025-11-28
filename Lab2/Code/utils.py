import torch
import numpy as np
import matplotlib.pyplot as plt

def dice_score(pred_mask, gt_mask, threshold=0.5, eps=1e-7):
    
    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = (pred_mask > threshold).float()

    intersection = (pred_mask * gt_mask).sum(dim=(1, 2, 3))
    union = pred_mask.sum(dim=(1, 2, 3)) + gt_mask.sum(dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()

def tensor_to_image(tensor):
    
    array = tensor.detach().cpu().numpy()
    if array.shape[0] == 1:
        array = array.squeeze(0)  # (1, H, W) → (H, W)
    else:
        array = np.moveaxis(array, 0, -1)  # (C, H, W) → (H, W, C)
    return array

    '''
    # Check if input is a NumPy array and convert to tensor if needed
    if isinstance(tensor, np.ndarray):
        array = tensor
    else:
        array = tensor.detach().cpu().numpy()

    # If the array has a single channel, squeeze the extra dimension (H, W)
    if array.shape[0] == 1:
        array = array.squeeze(0)  # (1, H, W) → (H, W)
    else:
        array = np.moveaxis(array, 0, -1)  # (C, H, W) → (H, W, C)
    return array'
    '''


def show_prediction(image_tensor, pred_mask_tensor, title="Prediction"):

    image = tensor_to_image(image_tensor)
    mask = tensor_to_image(pred_mask_tensor)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("Image")

    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(mask, alpha=0.4, cmap="Reds")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()


