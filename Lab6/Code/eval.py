import os
import json
import torch
from torchvision import transforms
from PIL import Image
from utils import evaluate_images
from dataset import labels_to_onehot
import warnings
warnings.filterwarnings("ignore")
def load_generated_images(image_folder):
    images = []
    for i in range(32):  # 32 張圖片
        path = os.path.join(image_folder, f"{i}.png")
        img = Image.open(path).convert("RGB")
        img_tensor = transforms.ToTensor()(img)  # 將圖片轉成(3, 64, 64)，範圍[0, 1]
        images.append(img_tensor)
    return torch.stack(images)  # 32張圖片變成一batch(32, 3, 64, 64)

def load_labels(json_file):
    with open(json_file, 'r') as f:
        label_list = json.load(f)
    return torch.stack([labels_to_onehot(l) for l in label_list]) # 轉成 one-hot encoding

def main():
    for name in ['test', 'new_test']:
        print(f"== Evaluating {name}.json ==")
        images = load_generated_images(f'images/{name}')
        labels = load_labels(f'{name}.json')
        evaluate_images(images, labels) # 計算正確率

if __name__ == '__main__':
    main()
