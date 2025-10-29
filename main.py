# main.py
import numpy as np
from ARdL_lib.ardl.train_cnn import train_cnn, predict

# -------------------------
# Dataset
# -------------------------
data = np.load("mnist.npz")
X_train, y_train = data['x_train'], data['y_train']
X_test, y_test = data['x_test'], data['y_test']

# Görselleri normalize et
X_train = X_train / 255.0
X_test = X_test / 255.0

# -------------------------
# Küçük alt küme ve sadece ilk 3 sınıf
# -------------------------
classes_to_use = [0, 1] # sadece 0,1 

train_mask = np.isin(y_train, classes_to_use)
test_mask = np.isin(y_test, classes_to_use)

X_train_small = X_train[train_mask][:25]
y_train_small = y_train[train_mask][:25]

X_test_small = X_test[test_mask][:5]
y_test_small = y_test[test_mask][:5]

# One-hot encoding
num_classes = 2  # sadece 0,1
y_train_small_onehot = np.eye(num_classes)[y_train_small]
y_test_small_onehot = np.eye(num_classes)[y_test_small]

# -------------------------
# Conv katmanı 
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
    }
]

# -------------------------
# Modeli eğit
# -------------------------
epochs = 10
lr = 0.01

conv_layers, W_dense, b_dense, history = train_cnn(
    X_train_small,
    y_train_small_onehot,
    conv_layers,
    lr=lr,
    epochs=epochs
)

# -------------------------
# Tahmin
# -------------------------
print("\nTahminler:")
for i in range(len(X_test_small)):
    y_pred = predict(X_test_small[i], conv_layers, W_dense, b_dense)
    print(f"Örnek {i+1}: Tahmin = {np.argmax(y_pred)}, Gerçek = {y_test_small[i]}")
