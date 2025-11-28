import torch
from torchvision import transforms
from evaluator import evaluation_model

def denormalize(x):
    # 原x: (batch, 3, 64, 64), [-1, 1]
    return (x.clamp(-1, 1) + 1) / 2 # 將影像資料從[-1, 1]轉回[0, 1]區間（RGB顯示用）

def evaluate_images(images, labels):
    # images: (batch, 3, 64, 64), labels: (batch, 24)
    norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 讓圖片從[0, 1]轉為[−1, 1]
    images = norm(images)
    evaluator = evaluation_model() # 呼叫evaluator
    acc = evaluator.eval(images.cuda(), labels.cuda())
    print(f"Accuracy: {acc:.3f}")
    return acc
