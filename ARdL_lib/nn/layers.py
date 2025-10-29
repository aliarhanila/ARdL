import numpy as np

# -------------------------
# Activations
# -------------------------
def ReLu(x):
    return np.maximum(0, x)

def ReLu_grad(x):
    return (x > 0).astype(float)

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Sigmoid_grad(x):
    s = Sigmoid(x)
    return s * (1 - s)

def Softmax(x):
    exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)

# -------------------------
# Loss
# -------------------------
def categorical_crossentropy(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

# -------------------------
# Dense Network Weights and Biases
# -------------------------
def Weights_Bias(input_dim, hidden_dims, output_dim):
    dims = [input_dim] + hidden_dims + [output_dim]
    weights, biases = [], []
    for i in range(len(dims)-1):
        w = np.random.randn(dims[i], dims[i+1]) * 0.1
        b = np.zeros(dims[i+1])
        weights.append(w)
        biases.append(b)
    return weights, biases

# -------------------------
# Forward
# -------------------------
def Forward(X, weights, biases, activations, output_activation='sigmoid'):
    a = X
    zs = []
    acts = [a]

    for i in range(len(weights)-1):
        z = a.dot(weights[i]) + biases[i]
        a = activations[i](z)
        zs.append(z)
        acts.append(a)

    z_out = a.dot(weights[-1]) + biases[-1]
    if output_activation == 'sigmoid':
        a = Sigmoid(z_out)
    elif output_activation == 'softmax':
        a = Softmax(z_out)
    zs.append(z_out)
    acts.append(a)
    return acts, zs

# -------------------------
# Backprop
# -------------------------
def BackProp(weights, biases, acts, zs, activation_grad, y, lr=0.1, output_activation='sigmoid'):
    L = len(weights)
    deltas = [0]*L

    if output_activation == 'sigmoid':
        deltas[-1] = (acts[-1] - y) * Sigmoid_grad(zs[-1])
    elif output_activation == 'softmax':
        deltas[-1] = acts[-1] - y

    for l in range(L-2, -1, -1):
        deltas[l] = deltas[l+1].dot(weights[l+1].T) * activation_grad(zs[l])

    for l in range(L):
        dW = acts[l].T.dot(deltas[l])
        db = np.sum(deltas[l], axis=0)
        weights[l] -= lr * dW
        biases[l] -= lr * db

    return weights, biases

# -------------------------
# Conv ve Pooling
# -------------------------
def conv2d(X, K, stride=1):
    X_h, X_w = X.shape
    K_h, K_w = K.shape
    out_h = (X_h - K_h)//stride + 1
    out_w = (X_w - K_w)//stride + 1
    Y = np.zeros((out_h, out_w))
    for i in range(0, X_h - K_h + 1, stride):
        for j in range(0, X_w - K_w + 1, stride):
            Y[i//stride, j//stride] = np.sum(X[i:i+K_h, j:j+K_w] * K)
    return Y

def conv2d_backward(dY, X, K, stride=1):
    H, W = X.shape
    kH, kW = K.shape
    dK = np.zeros_like(K)
    dX = np.zeros_like(X)

    for i in range(0, H - kH + 1, stride):
        for j in range(0, W - kW + 1, stride):
            region = X[i:i+kH, j:j+kW]
            dK += dY[i//stride, j//stride] * region
            dX[i:i+kH, j:j+kW] += dY[i//stride, j//stride] * K

    return dX, dK

def max_pool2d(X, size=2, stride=2):
    X_h, X_w = X.shape
    out_h = (X_h - size)//stride + 1
    out_w = (X_w - size)//stride + 1
    Y = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = X[i*stride:i*stride+size, j*stride:j*stride+size]
            Y[i,j] = np.max(patch)
    return Y

def max_pool2d_backward(dY, X, size=2, stride=2):
    dX = np.zeros_like(X)
    out_h, out_w = dY.shape
    for i in range(out_h):
        for j in range(out_w):
            patch = X[i*stride:i*stride+size, j*stride:j*stride+size]
            idx = np.unravel_index(np.argmax(patch), patch.shape)
            dX[i*stride+idx[0], j*stride+idx[1]] = dY[i,j]
    return dX
# -------------------------
# Helper: img2col
# -------------------------
def im2col(X, kernel_size, stride=1):
    """
    X: (H, W) input image
    kernel_size: int
    stride: int
    return: (out_h*out_w, kernel_size*kernel_size) matrix
    """
    H, W = X.shape
    out_h = (H - kernel_size) // stride + 1
    out_w = (W - kernel_size) // stride + 1

    cols = []
    for i in range(0, out_h*stride, stride):
        for j in range(0, out_w*stride, stride):
            patch = X[i:i+kernel_size, j:j+kernel_size].ravel()
            cols.append(patch)
    return np.array(cols)
#---------------------
#Deneysel Optimizasyon
#--------------------
def conv2d_vec(X, kernel, stride=1):
    """
    X: (H, W)
    kernel: (kH, kW)
    """
    kH, kW = kernel.shape
    X_col = im2col(X, kH, stride)  # (out_h*out_w, kH*kW)
    kernel_col = kernel.ravel()    # (kH*kW,)
    out = X_col.dot(kernel_col)    # matris çarpımı
    out_h = (X.shape[0]-kH)//stride + 1
    out_w = (X.shape[1]-kW)//stride + 1
    return out.reshape(out_h, out_w)

def max_pool2d_vec(X, size=2, stride=2):
    H, W = X.shape
    out_h = (H - size) // stride + 1
    out_w = (W - size) // stride + 1

    cols = []
    for i in range(0, out_h*stride, stride):
        for j in range(0, out_w*stride, stride):
            patch = X[i:i+size, j:j+size]
            cols.append(np.max(patch))
    return np.array(cols).reshape(out_h, out_w)
