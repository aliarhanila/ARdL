import numpy as np
from ARdL_lib.ardl.train_cnn import train_cnn, predict


# Testing Mnist Dataset On CNN algoritm


# -------------------------
# Dataset
# -------------------------
data = np.load("mnist.npz")  # dosya adını kendi datasetinle değiştir
X_train, y_train = data['x_train'], data['y_train']
X_test, y_test = data['x_test'], data['y_test']

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# -------------------------
# Small subset and only first 5 classes
# -------------------------
classes_to_use = [0, 1 , 2 , 3 , 4 ]

train_mask = np.isin(y_train, classes_to_use)
test_mask = np.isin(y_test, classes_to_use)

X_train_small = X_train[train_mask][:500]
y_train_small = y_train[train_mask][:500]

X_test_small = X_test[test_mask][:100]
y_test_small = y_test[test_mask][:100]

# One-hot encoding
num_classes = 5
y_train_small_onehot = np.eye(num_classes)[y_train_small]
y_test_small_onehot = np.eye(num_classes)[y_test_small]

# -------------------------
# Conv layer
# -------------------------
conv_layers = [
    {
        "type": "conv",
        "kernel_size": 3,
        "activation": "relu",
        "stride": 1,
        "pool": 2,
        "pool_stride": 1,
        "kernel": np.random.randn(3,3)*0.1
    },
    {
        "type": "conv",
        "kernel_size": 3,
        "activation": "relu",
        "stride": 1,
        "pool": 2,
        "pool_stride": 1,
        "kernel": np.random.randn(3,3)*0.1
    }
]

# -------------------------
# Model Train
# -------------------------
epochs = 20
lr = 0.001

conv_layers, W_dense, b_dense, history = train_cnn(
    X_train_small,
    y_train_small_onehot,
    conv_layers,
    lr=lr,
    epochs=epochs,
    X_test=X_test_small,
    y_test=y_test_small_onehot,
    random_seed=10
)

# -------------------------
# Predicts
# -------------------------
print("\nTahminler:")
for i in range(len(X_test_small)):
    y_pred = predict(X_test_small[i], conv_layers, W_dense, b_dense)
    print(f"Örnek {i+1}: Tahmin = {np.argmax(y_pred)}, Gerçek = {y_test_small[i]}")

# -------------------------
# Model Save
# -------------------------
np.savez("cnn_model.npz", weights=W_dense, biases=b_dense)

