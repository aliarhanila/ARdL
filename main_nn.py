import numpy as np
from ARdL_lib.nn.train_mlp import train_nn

# One-hot encoding
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

np.random.seed(42)
X_train = np.random.rand(120, 4)
y_train = np.random.randint(0, 3, 120)
X_test = np.random.rand(30, 4)
y_test = np.random.randint(0, 3, 30)

y_train_oh = one_hot(y_train, 3)
y_test_oh = one_hot(y_test, 3)

# Model Train
weights, biases, history = train_nn(
    X_train, y_train_oh,
    hidden_dims=[16, 8],
    lr=0.01,
    epochs=300,
    output_activation='softmax',
    X_test=X_test, y_test=y_test_oh,
    verbose=True
)

# Sonuçlar
print("\n--- Training Completed ---")
print(f"Final Train Acc: {history['train_acc'][-1]*100:.2f}%")
print(f"Final Test Acc:  {history['test_acc'][-1]*100:.2f}%")
