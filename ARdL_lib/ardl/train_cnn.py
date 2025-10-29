import numpy as np
from ARdL_lib.nn.layers import ReLu, ReLu_grad, Softmax, categorical_crossentropy
from ARdL_lib.nn.layers import conv2d_vec, max_pool2d_vec , max_pool2d_backward ,conv2d_backward

# -------------------------
# Flatten after conv
# -------------------------
def flatten_after_conv(X, conv_layers):
    a = np.array(X, dtype=np.float64)
    for conv in conv_layers:
        a = conv2d_vec(a, conv['kernel'], stride=conv.get('stride',1))
        if conv.get('activation','relu') == 'relu':
            a = ReLu(a)
        if conv.get('pool', None):
            a = max_pool2d_vec(a, size=conv['pool'], stride=conv.get('pool_stride', conv.get('stride',1)))
    return a.ravel()

# -------------------------
# Predict
# -------------------------
def predict(X, conv_layers, W_dense, b_dense):
    a = np.array(X, dtype=np.float64)
    for conv in conv_layers:
        a = conv2d_vec(a, conv['kernel'], stride=conv.get('stride',1))
        if conv.get('activation','relu') == 'relu':
            a = ReLu(a)
        if conv.get('pool', None):
            a = max_pool2d_vec(a, size=conv['pool'], stride=conv.get('pool_stride', conv.get('stride',1)))
    a_flat = a.ravel()
    z = a_flat.dot(W_dense) + b_dense
    return Softmax(z)

# -------------------------
# CNN Train Function
# -------------------------
def train_cnn(X_train, y_train, conv_layers, lr=0.01, epochs=10, X_test=None, y_test=None,random_seed=42):
    np.random.seed(random_seed)
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

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for i in range(len(X_train)):
            x = X_train[i]
            y_true = y_train[i]

            # ---------- Forward ----------
            a = x
            conv_cache = []
            conv_outs = []
            for conv in conv_layers:
                z = conv2d_vec(a, conv['kernel'], stride=conv.get('stride',1))
                conv_cache.append(a.copy())
                conv_outs.append(z.copy())
                if conv.get('activation','relu') == 'relu':
                    z = ReLu(z)
                if conv.get('pool', None):
                    z = max_pool2d_vec(z, size=conv['pool'], stride=conv.get('pool_stride', conv.get('stride',1)))
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
                    # Eğer max_pool2d_vec için backward yazmadıysan önce backward fonksiyonu yazman gerekir
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

                # Top layer delta
                delta_conv, _ = conv2d_backward(delta_conv, a_prev, conv['kernel'], stride=conv.get('stride',1))

        train_loss = total_loss / len(X_train)
        train_acc = correct / len(X_train)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # ---------- Test ----------
        if X_test is not None and y_test is not None:
            test_loss = 0
            test_correct = 0
            for i in range(len(X_test)):
                y_pred = predict(X_test[i], conv_layers, W_dense, b_dense)
                test_loss += categorical_crossentropy(y_test[i].reshape(1,-1), y_pred.reshape(1,-1))
                if np.argmax(y_pred) == np.argmax(y_test[i]):
                    test_correct += 1
            test_loss /= len(X_test)
            test_acc = test_correct / len(X_test)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}% | "
                  f"Test Loss={test_loss:.4f}, Test Acc={test_acc*100:.2f}%")
        else:
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}%")

    return conv_layers, W_dense, b_dense, history
