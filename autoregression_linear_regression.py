# autoregression with linear regression

import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np 

def rnn_sin_wave(tau = 10, num_iter = 150, noise = 0.1):
    T = 1000
    time = torch.arange(1, T + 1, dtype=torch.float32)
    x = torch.sin(0.01 * time) + torch.randn(T) * noise

    num_train = 600

    features = [x[i: T-tau+i] for i in range(tau)]
    X = torch.stack(features, 1)
    X_b = torch.cat((torch.ones(len(X), 1), X), 1)
    y = x[tau:].reshape((-1, 1))
    Xtrain = X_b[:num_train]
    ytrain = y[:num_train]
    wb_best = torch.linalg.inv(Xtrain.T.matmul(Xtrain)).matmul(Xtrain.T).matmul(ytrain) 

    # Initialize plot
    plt.figure(figsize=(12, 6))
    plt.plot(time, x, label='Original Data', color='blue')

    y_pred_list = []

    pos = torch.randint(T - tau - num_iter, (1,)).item()
    pos_start = pos

    for _ in range(num_iter):
        prompt = x[pos:pos + tau]
        extra1 = torch.ones(1,)
        x_b = torch.cat((extra1, prompt))
        y_pred = x_b.matmul(wb_best)
        y_pred_list.append(y_pred)

        #print(y_pred)
        prompt = torch.cat((prompt[1:], y_pred))
        x_b = torch.cat((extra1, prompt))
        #print(x_b)

        pos += 1

    plt.plot(time[pos_start+tau:pos_start+num_iter+tau], y_pred_list, color='red', label='Predicted Line')

    mse = torch.mean((x[pos_start+tau:pos_start+num_iter+tau] - torch.stack(y_pred_list))**2).item()
    print(mse)
    plt.title(f'\nTau = {tau}\nNum pred = {num_iter}\nNoise = {noise}\nMSE = {mse}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(rf"C:/DL/tau_{tau}_iter_{num_iter}_noise_{noise}.png", dpi=300)
    #plt.show()

if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Run rnn_sin_wave function with specified arguments.")

    # Add command-line arguments
    parser.add_argument('--tau', type=int, default=10, help='Value for tau')
    parser.add_argument('--num_iter', type=int, default=150, help='Value for num_iter')
    parser.add_argument('--noise', type=float, default=0.1, help='Value for noise')

    # Parse the command-line arguments
    args = parser.parse_args()

    rnn_sin_wave(tau=args.tau, num_iter=args.num_iter, noise=args.noise)