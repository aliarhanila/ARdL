# ardl/train_nn.py
import numpy as np
from ARdL_lib.nn.layers import Weights_Bias, Forward, BackProp, ReLu, ReLu_grad, Sigmoid, Softmax

def train_nn(X, y, hidden_dims=[8, 8], lr=0.1, epochs=1000, output_activation='sigmoid', verbose=True):
    input_dim = X.shape[1]
    output_dim = y.shape[1]

    weights, biases = Weights_Bias(input_dim, hidden_dims, output_dim)
    activations = [ReLu for _ in hidden_dims]

    for epoch in range(epochs):
        acts, zs = Forward(X, weights, biases, activations, output_activation=output_activation)
        weights, biases = BackProp(weights, biases, acts, zs, ReLu_grad, y, lr=lr, output_activation=output_activation)

        if verbose and epoch % 100 == 0:
            loss = np.mean((acts[-1] - y) ** 2)
            print(f"Epoch {epoch}: Loss = {loss:.6f}")

    return weights, biases
