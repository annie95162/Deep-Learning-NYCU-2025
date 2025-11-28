import numpy as np
from layers import *
from utils import *
from optimizers import *
from nn import *

def train_test_model(model, x, y, epochs=20000, epochs_print=1000):
    losses = model.train(x, y, epochs, epochs_print)  
    predictions = model.predict(x)  
    binary_preds = (predictions > 0.5).astype(int)  
    accuracy = np.mean(y == binary_preds)  

    for i, (yi, pred_y) in enumerate(zip(y, binary_preds)):
        print(f"Iter: {i} |\t Ground truth: {yi} |\t Predict: {pred_y}")
    print("Final Loss: {:.5f}, Accuracy: {:.2f}%".format(losses[-1], accuracy * 100))  

    plot_loss_curve(losses)  
    show_result(x, y, binary_preds)


def main():
    np.random.seed(16)

    lr = 0.1  
    optimizer = SGD  
    layer = Affine  
    activation = Sigmoid 

    print("Training on Linear Dataset")
    x, y = generate_linear(n=100)

    ### 執行方法 ###

    model = NN(2, [10, 10], 1, lr, optimizer, layer, activation)  
    # model=NN_no_activ(2, [10, 10], 1, lr=0.01, optimizer=SGD)

    ### 上面呼叫NN或NN_no_activ ###

    train_test_model(model, x, y)

    print("Training on XOR Dataset")
    x, y = generate_XOR_easy()

    ### 執行方法 ###

    model_xor = NN(2, [10, 10], 1, lr, optimizer, layer, activation)  
    # model_xor=NN_no_activ(2, [10, 10], 1, lr=0.01, optimizer=SGD)

    ### 上面呼叫NN或NN_no_activ ###
    
    train_test_model(model_xor, x, y)

if __name__ == "__main__":
    main()