import numpy as np
from ARdL_lib.nn.layers import Weights_Bias, Forward, BackProp, ReLu, ReLu_grad, Sigmoid, Softmax

def train_nn(X_train, y_train, hidden_dims=[8, 8], lr=0.1, epochs=1000,
             output_activation='sigmoid', X_test=None, y_test=None, verbose=True):
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    # Create a weights and Biases
    weights, biases = Weights_Bias(input_dim, hidden_dims, output_dim)
    activations = [ReLu for _ in hidden_dims]

    # History lists 
    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": []
    }

    for epoch in range(epochs):
        # ---------- Forward ----------
        acts, zs = Forward(X_train, weights, biases, activations, output_activation=output_activation)
        y_pred = acts[-1]

        # ---------- Loss & Accuracy ----------
        train_loss = np.mean((y_pred - y_train) ** 2)
        train_acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_train, axis=1))

        # ---------- Backprop ----------
        weights, biases = BackProp(weights, biases, acts, zs, ReLu_grad, y_train,
                                   lr=lr, output_activation=output_activation)

        # ---------- Test Data ----------
        if X_test is not None and y_test is not None:
            acts_test, _ = Forward(X_test, weights, biases, activations, output_activation=output_activation)
            y_pred_test = acts_test[-1]
            test_loss = np.mean((y_pred_test - y_test) ** 2)
            test_acc = np.mean(np.argmax(y_pred_test, axis=1) == np.argmax(y_test, axis=1))
        else:
            test_loss, test_acc = None, None

        # ---------- History Update ----------
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        # ---------- Print ----------
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            if test_loss is not None:
                print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}% | "
                      f"Test Loss={test_loss:.4f}, Test Acc={test_acc*100:.2f}%")
            else:
                print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}%")

    return weights, biases, history
