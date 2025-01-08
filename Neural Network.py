import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size)
        self.B1 = np.random.randn(1, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.B2 = np.random.randn(1, output_size)

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return Z > 0

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / exp_Z.sum(axis=1, keepdims=True)

    def forward(self, X):
        # First layer
        self.Z1 = np.dot(X, self.W1) + self.B1
        self.A1 = self.relu(self.Z1)

        # Second layer
        self.Z2 = np.dot(self.A1, self.W2) + self.B2
        self.A2 = self.softmax(self.Z2)

        return self.A2

    def loss(self, y_true, y_pred):
        # Cross-Entropy Loss
        n_samples = y_true.shape[0]
        # Ensure y_true is one-hot encoded (for classification)
        y_ohe = np.zeros_like(y_pred)
        y_ohe[np.arange(n_samples), y_true] = 1  # Set the correct class index to 1

        # Cross-entropy loss formula
        log_probs = -np.log(y_pred[range(n_samples), y_true])
        loss = np.sum(log_probs) / n_samples
        return loss

    def backpropagate(self, X, y_true):
        # One-hot encode the output
        n_samples = X.shape[0]
        y_ohe = np.zeros_like(self.A2)
        y_ohe[np.arange(n_samples), y_true] = 1

        # Calculate gradients for W2 and B2
        dZ2 = self.A2 - y_ohe
        dW2 = np.dot(self.A1.T, dZ2) / n_samples
        dB2 = np.sum(dZ2, axis=0, keepdims=True) / n_samples

        # Backpropagate to the first layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / n_samples
        dB1 = np.sum(dZ1, axis=0, keepdims=True) / n_samples

        # Update weights and biases
        learning_rate = 0.05
        self.W1 -= learning_rate * dW1
        self.B1 -= learning_rate * dB1
        self.W2 -= learning_rate * dW2
        self.B2 -= learning_rate * dB2

    def train(self, X, y, epochs=100):
        loss_history = []  # List to store loss at each epoch
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.loss(y, y_pred)
            loss_history.append(loss)  # Store loss for this epoch
            self.backpropagate(X, y)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')
        return loss_history  # Return loss history

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)  # Return class with highest probability



# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data (normalize and flatten)
X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32) / 255.0

# Initialize the neural network
input_size = 28 * 28  # Each image is 28x28 pixels
hidden_size = 256  # Number of neurons in the hidden layer
output_size = 10  # There are 10 classes (digits 0-9)

nn = NeuralNetwork(input_size, hidden_size, output_size)

# Train the neural network and record loss history
loss_history = nn.train(X_train, y_train, epochs=100)

# Plot loss over epochs
plt.plot(range(100), loss_history)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.show()

# Make predictions and calculate accuracy
y_pred = nn.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy on test data: {accuracy * 100:.2f}%')
