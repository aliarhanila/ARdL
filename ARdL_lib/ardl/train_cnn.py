import numpy as np
from ARdL_lib.nn.layers import ReLu, ReLu_grad, Softmax, categorical_crossentropy
from ARdL_lib.nn.layers import conv2d, conv2d_backward, max_pool2d, max_pool2d_backward

# -------------------------
# Helper: Conv + Flatten
# -------------------------
def flatten_after_conv(X, conv_layers):
    a = np.array(X, dtype=np.float64)
    for conv in conv_layers:
        a = np.array(conv2d(a, conv['kernel'], stride=conv.get('stride',1)), dtype=np.float64)
        if conv.get('activation','relu') == 'relu':
            a = ReLu(a)
        if conv.get('pool', None):
            a = np.array(max_pool2d(a, size=conv['pool']), dtype=np.float64)
    return a.ravel()

# -------------------------
# Predict
# -------------------------
def predict(X, conv_layers, W_dense, b_dense):
    a = np.array(X, dtype=np.float64)
    for conv in conv_layers:
        a = np.array(conv2d(a, conv['kernel'], stride=conv.get('stride',1)), dtype=np.float64)
        if conv.get('activation','relu') == 'relu':
            a = ReLu(a)
        if conv.get('pool', None):
            a = np.array(max_pool2d(a, size=conv['pool']), dtype=np.float64)
    a_flat = a.ravel()
    z = a_flat.dot(W_dense) + b_dense
    return Softmax(z)

# -------------------------
# CNN Train Function (Conv + Dense Backprop)
# -------------------------
def train_cnn(X_train, y_train, conv_layers, lr=0.01, epochs=10):
    # Conv kernel initialize
    for conv in conv_layers:
        if 'kernel' not in conv or conv['kernel'] is None:
            k_size = conv.get('kernel_size', 3)
            conv['kernel'] = np.random.randn(k_size, k_size) * 0.1

    # Dense Size
    sample_out = flatten_after_conv(X_train[0], conv_layers)
    flatten_dim = sample_out.shape[0]
    W_dense = np.random.randn(flatten_dim, y_train.shape[1]) * 0.1
    b_dense = np.zeros(y_train.shape[1])

    history = {'train_loss': [], 'train_acc': []}

    for epoch in range(epochs):
        total_loss = 0
        correct = 0

        for i in range(len(X_train)):
            x = X_train[i]
            y_true = y_train[i]

            # ---------- Forward Pass ----------
            a = x
            conv_cache = []
            conv_outs = []
            for conv in conv_layers:
                z = conv2d(a, conv['kernel'], stride=conv.get('stride',1))
                conv_cache.append(a.copy())
                conv_outs.append(z.copy())
                if conv.get('activation','relu') == 'relu':
                    z = ReLu(z)
                if conv.get('pool', None):
                    z = max_pool2d(z, size=conv['pool'])
                a = z

            a_flat = a.ravel()
            z_out = a_flat.dot(W_dense) + b_dense
            y_pred = Softmax(z_out)

            # ---------- Loss & Accuracy ----------
            total_loss += categorical_crossentropy(y_true.reshape(1,-1), y_pred.reshape(1,-1))
            if np.argmax(y_pred) == np.argmax(y_true):
                correct += 1

            # ---------- Backprop Dense ----------
            delta_dense = y_pred - y_true
            dW_dense = np.outer(a_flat, delta_dense)
            db_dense = delta_dense
            W_dense -= lr * dW_dense
            b_dense -= lr * db_dense

            # ---------- Backprop Conv ----------
            delta_conv = delta_dense.dot(W_dense.T).reshape(a.shape)
            for idx in reversed(range(len(conv_layers))):
                conv = conv_layers[idx]
                a_prev = conv_cache[idx]

                # Pool backward
                if conv.get('pool', None):
                    delta_conv = max_pool2d_backward(
                        delta_conv,
                        conv_outs[idx],
                        size=conv['pool'],
                        stride=conv.get('pool_stride', conv.get('stride',1))
                    )

                # ReLU backward
                if conv.get('activation','relu') == 'relu':
                    delta_conv *= ReLu_grad(conv_outs[idx])

                # Conv kernel update
                _, dK = conv2d_backward(delta_conv, a_prev, conv['kernel'], stride=conv.get('stride',1))
                conv['kernel'] -= lr * dK

                # top layer  delta
                delta_conv, _ = conv2d_backward(delta_conv, a_prev, conv['kernel'], stride=conv.get('stride',1))

        train_loss = total_loss / len(X_train)
        train_acc = correct / len(X_train)
        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc*100:.2f}%")
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

    return conv_layers, W_dense, b_dense, history
