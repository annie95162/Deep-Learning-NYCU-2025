import numpy as np
from typing import *
from layers import *
from optimizers import *

np.random.seed(16)  # 固定隨機種子，確保結果可重現

class NN():
    def __init__(self, input_size: int, hidden_size: List[int], output_size: int, 
                 lr: float=0.01, op=SGD, layer=Affine, activation=Sigmoid):

        self.input_size = input_size  # 輸入層大小
        self.hidden_size = hidden_size  # 隱藏層大小
        self.output_size = output_size  # 輸出層大小
        self.lr = lr  # 學習率
        self.opti = op(lr)  # 初始化優化器
        self.layer = layer  # 隱藏層類型
        self.activation = activation  # 預設 Sigmoid

        # 初始化權重與偏差
        self.params = {}
        layer_sizes = [input_size] + hidden_size + [output_size]
        for i in range(1, len(layer_sizes)):  
            x, y = layer_sizes[i - 1], layer_sizes[i]  # 取得前一層與當層的神經元數量
            self.params[f'W{i}'] = np.random.randn(x, y)  # 隨機初始化權重
            self.params[f'b{i}'] = np.zeros(y)  # 初始化b為0

        self.layers = {}
        for i in range(1, len(hidden_size) + 2):  
            self.layers[f'Layer{i}'] = self.layer(self.params[f'W{i}'], self.params[f'b{i}'])  # 加入線性層
            self.layers[f'Activation{i}'] = activation()


    def forward(self, x):

        current = x  
        for layer in self.layers.values():
            out = layer.forward(current)
            current = out  
        self.pred_y = current
        return current

    
    def backward(self, y):
        out = y
        for layer in reversed(self.layers.values()):
            out = layer.backward(out)
        return out
    
    def predict(self, x):

        return self.forward(x)
    
 
    def train(self, data_x, data_y, num_epochs=10000, print_interval=500):
        losses = []
        for epoch in range(1, num_epochs + 1):
            loss = np.mean((self.forward(data_x) - data_y) ** 2)  # 計算 MSE
            self.backward(2 * (self.pred_y - data_y) / data_y.shape[0])  # 計算梯度
            self.update()  # 更新權重
            
            losses.append(loss)  # 存 loss
            if epoch % print_interval == 0: print(f'Epoch {epoch}: Loss = {loss:.5f}')
        
        # print(f'Final Loss: {losses[-1]:.5f}')
        return losses


   
    def update(self):

        grads = {}
        lyrs = [lyr for lyr in self.layers.values() if isinstance(lyr, self.layer)]
        
        for i, lyr in enumerate(lyrs, 1):
            grads[f'W{i}'] = lyr.grad_weig
            grads[f'b{i}'] = lyr.grad_b
        
        self.opti.update(self.params, grads)  # 使用優化器更新權重

class NN_no_activ():
    def __init__(self, input_size: int, hidden_size: List[int], output_size: int, 
                 lr: float=0.01, optimizer=SGD, layer=Affine):
        
        self.input_size = input_size  # 輸入層大小
        self.hidden_size = hidden_size  # 隱藏層大小
        self.output_size = output_size  # 輸出層大小
        self.lr = lr  # 學習率
        self.opti = optimizer(lr)  # 優化器
        self.layer = layer  # 隱藏層類型（沒有activation）
        
        # 初始化權重與偏差
        self.params = {}
        layer_sizes = [input_size] + hidden_size + [output_size]
        for i in range(1, len(layer_sizes)):
            x, y = layer_sizes[i - 1], layer_sizes[i]
            self.params[f'W{i}'] = np.random.randn(x, y)  # 隨機初始化權重
            self.params[f'b{i}'] = np.zeros(y)  # 初始化偏差為0

        self.layers = {}
        for i in range(1, len(hidden_size) + 2):
            self.layers[f'Layer{i}'] = self.layer(self.params[f'W{i}'], self.params[f'b{i}'])

    def forward(self, x):
        """前向傳播，沒有激活函數"""
        current = x
        for layer in self.layers.values():
            current = layer.forward(current)  # 每層進行線性運算
        self.pred_y = current  # 輸出預測結果
        return current
    
    def backward(self, y):
        """反向傳播，計算梯度"""
        dout = y
        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)  # 每層進行反向傳播
        return dout

    def predict(self, x):
        """預測結果"""
        return self.forward(x)
    
    def train(self, data_x, data_y, num_epochs=10000, print_interval=500):
        """訓練模型"""
        losses = []
        for epoch in range(1, num_epochs + 1):
            loss = np.mean((self.forward(data_x) - data_y) ** 2)  # 計算均方誤差（MSE）
            
            self.backward(2 * (self.pred_y - data_y) / data_y.shape[0])  # 計算梯度
            # self.backward(self.pred_y - data_y)  # 計算梯度 # no activation linear fail
            self.update()  # 更新權重
            
            losses.append(loss)  # 記錄每次損失
            if epoch % print_interval == 0: 
                
                print(f'Epoch {epoch}: Loss = {loss:.5f}')
        
        print(f'Final Loss: {losses[-1]:.5f}')  # 顯示最終損失
        return losses
    
    def update(self):
        """更新權重與偏差"""
        grads = {}
        layers = [layer for layer in self.layers.values() if isinstance(layer, self.layer)]
        
        for i, layer in enumerate(layers, 1):
            grads[f'W{i}'] = layer.grad_weig
            grads[f'b{i}'] = layer.grad_b
        
        self.opti.update(self.params, grads)  # 使用優化器更新權重
