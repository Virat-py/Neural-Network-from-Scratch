# Neural Network for MNIST Digit Classification

This project implements a simple feedforward neural network from scratch to classify the MNIST dataset. The MNIST dataset consists of 28x28 pixel grayscale images of handwritten digits (0-9). The network is trained using backpropagation with gradient descent and uses a ReLU activation function in the hidden layer and a softmax activation function in the output layer.


## Model Architecture

- **Input Layer**: 784 units (28x28 pixels flattened).
- **Hidden Layer**: 256 units with ReLU activation.
- **Output Layer**: 10 units (one for each digit) with softmax activation.


## Training the Model

The model is trained using gradient descent and backpropagation. The loss is calculated at each epoch using cross-entropy loss, and the model parameters are updated using the gradients from backpropagation.


## Results

Once trained, the model can predict the class of digits in the MNIST test set. The model's accuracy on the test data after 100 epochs of training is **80.17%**.

## Technologies Used

- **Python**: The programming language used for the neural network implementation.
- **NumPy**: For matrix operations, numerical computations, and handling of arrays.
- **Matplotlib**: For plotting the loss curve over epochs.
- **TensorFlow Keras Dataset**: To load the MNIST dataset.
