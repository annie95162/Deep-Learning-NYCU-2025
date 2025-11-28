import numpy as np
import matplotlib.pyplot as plt

# 生成線性可分
def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))  # 生成n個 (x, y) 點，範圍在 [0,1]
    inputs = []  # 存放輸入數據
    labels = []  # 存放標籤數據
    
    for pt in pts:
        inputs.append([pt[0], pt[1]])  # 存入 (x, y) 值
        distance = (pt[0] - pt[1]) / 1.414  # 計算與y=x直線距離
        if pt[0] > pt[1]:  # 如果 x > y，標記為 0
            labels.append(0)
        else:  # 否則標記為 1
            labels.append(1)
    
    return np.array(inputs), np.array(labels).reshape(n, 1)  # 返回輸入和標籤

# 生成XOR
def generate_XOR_easy():
    inputs = []  # 存放輸入數據
    labels = []  # 存放標籤數據
    
    for i in range(11):  # 生成11組數據
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)
        if 0.1 * i == 0.5:
            continue  # 略過(0.5, 0.5)
        inputs.append([0.1 * i, 1 - 0.1 * i])  # (x, 1-x)為 1
        labels.append(1)
    
    return np.array(inputs), np.array(labels).reshape(21, 1)  # 返回輸入和標籤

# 顯示結果
def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground Truth')
    for i in range(len(y)):
        plt.plot(x[i, 0], x[i, 1], 'ro' if y[i] == 0 else 'bo')
    
    plt.subplot(1, 2, 2)
    plt.title('Prediction')
    for i in range(len(pred_y)):
        plt.plot(x[i, 0], x[i, 1], 'ro' if pred_y[i] == 0 else 'bo')
    
    plt.show()

# 顯示 Loss 變化曲線
def plot_loss_curve(loss):
    plt.plot(loss)
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
