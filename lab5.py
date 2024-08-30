import numpy as np

# Define the input data and output labels
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)  # Input data
y = np.array(([92], [86], [89]), dtype=float)  # Output labels

# Normalize the data (scaling between 0 and 1)
X = X / np.amax(X, axis=0)
y = y / 100  # Scaling output to 0-1 range

# Activation function: sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Parameters for the neural network
epochs = 5000  # Number of training iterations
learning_rate = 0.1
input_layer_neurons = 2
hidden_layer_neurons = 3
output_neurons = 1

# Initialize weights and biases
weights_input_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_layer_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

# Training the neural network
for epoch in range(epochs):
    # Forward propagation
    # Input to hidden layer
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_input += bias_hidden
    hidden_layer_activations = sigmoid(hidden_layer_input)

    # Hidden to output layer
    output_layer_input = np.dot(hidden_layer_activations, weights_hidden_output)
    output_layer_input += bias_output
    predicted_output = sigmoid(output_layer_input)

    # Backpropagation
    # Calculate error
    error = y - predicted_output
    
    # Compute gradients
    output_gradient = sigmoid_derivative(predicted_output)
    delta_output = error * output_gradient
    
    hidden_layer_error = delta_output.dot(weights_hidden_output.T)
    hidden_layer_gradient = sigmoid_derivative(hidden_layer_activations)
    delta_hidden_layer = hidden_layer_error * hidden_layer_gradient

    # Update weights and biases
    weights_hidden_output += hidden_layer_activations.T.dot(delta_output) * learning_rate
    weights_input_hidden += X.T.dot(delta_hidden_layer) * learning_rate

# Output after training
print("Input Data: \n", X)
print("Actual Output: \n", y)
print("Predicted Output: \n", predicted_output)


