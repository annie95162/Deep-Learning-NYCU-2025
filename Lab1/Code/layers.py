import numpy as np

# Sigmoid
class Sigmoid:
    def __init__(self):
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        exp_neg_x = np.exp(-x)  # 先計算-x的指數
        self.cache = 1 / (1 + exp_neg_x)  # 計算Sigmoid值
        return self.cache

    def backward(self, dout: np.ndarray) -> np.ndarray:
        grad = self.cache * (1 - self.cache)  # 計算Sigmoid導數
        return dout * grad  # 反向傳播梯度

# ReLU

class ReLU():
    def __init__(self):
        self.mask = None  # 儲存輸入值的狀態

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0  # 生成遮罩，>0=True
        return x * self.mask  # <=0 的會變成 0

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.mask  # 只有>0保留梯度，其餘變0

# Affine，線性變換Wx + b
class Affine:
    def __init__(self, weig: np.ndarray, b: np.ndarray):
        self.weig = weig  # 權重
        self.b = b.reshape(1, -1)  # 偏差
        self.x_data = None 
        self.grad_weig = None  # 權重梯度
        self.grad_b = None  # 偏差梯度

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_data = x  # 儲存輸入數據
        return x @ self.weig + self.b  # 使用@計算矩陣乘法

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad_in = grad_out @ self.weig.T  # 計算輸入梯度
        self.grad_weig = self.x_data.T @ grad_out  # 計算權重梯度
        self.grad_b = grad_out.sum(axis=0)  # 計算偏差梯度
        return grad_in

