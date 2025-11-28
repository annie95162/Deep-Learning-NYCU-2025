import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

with open('objects.json', 'r') as f:
    OBJ2IDX = json.load(f) # { "red cube": 0, "blue sphere": 1}
IDX2OBJ = {v: k for k, v in OBJ2IDX.items()} # { 0: "red cube", 1: "blue sphere"}
NUM_CLASSES = len(OBJ2IDX)

def labels_to_onehot(label_list): # one-hot encoding
    onehot = torch.zeros(NUM_CLASSES)
    for obj in label_list:
        onehot[OBJ2IDX[obj]] = 1
    return onehot

class CLEVRDataset(Dataset):
    def __init__(self, json_path, img_folder=None, transform=None):
        with open(json_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict): # train的格式
            self.samples = list(data.items())
        else:
            self.samples = data # test的格式
        self.img_folder = img_folder
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if isinstance(self.samples[idx], tuple): # train
            img_name, labels = self.samples[idx]
            img_path = os.path.join(self.img_folder, img_name)
            img = Image.open(img_path).convert('RGB')
        else:
            labels = self.samples[idx] # test
            img = Image.new('RGB', (64, 64), (0, 0, 0))  # 黑色空白圖像
        if self.transform:
            img = self.transform(img)
        label_onehot = labels_to_onehot(labels)
        return img, label_onehot
