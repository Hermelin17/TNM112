import numpy as np
import data_generator

# Different activations functions
def activation(x, activation):
    
    # TODO: specify the different activation functions
    # 'activation' could be: 'linear', 'relu', 'sigmoid', or 'softmax'
    if activation == 'linear':
        return x
    elif activation == 'sigmoid':
        return 1.0 / (1.0 + np.exp(-x))
    elif activation == 'relu':
        return np.maximum(0.0, x)
    elif activation == 'softmax':
        # x will be a column vector (K, 1); normalize to sum = 1
        ex = np.exp(x)
        return ex / np.sum(ex)
    else:
        raise Exception("Activation function is not valid", activation) 

#-------------------------------
# Our own implementation of an MLP
#-------------------------------
class MLP:
    def __init__(
        self,
        dataset,         # DataGenerator
    ):
        self.dataset = dataset

    # Set up the MLP from provided weights and biases
    def setup_model(
        self,
        W,                   # List of weight matrices
        b,                   # List of bias vectors
        activation='linear'  # Activation function of layers
    ):
        self.activation = activation

        # TODO: specify the number of hidden layers based on the length of the provided lists
        # All layers except the last one are hidden
        self.hidden_layers = len(W) - 1

        self.W = W
        self.b = b

        # TODO: specify the total number of weights in the model (both weight matrices and bias vectors)
        self.N = sum(w.size for w in W) + sum(bi.size for bi in b)

        print('Number of hidden layers: ', self.hidden_layers)
        print('Number of model weights: ', self.N)

    # Feed-forward through the MLP
    def feedforward(
        self,
        x      # Input data points, shape (N, D)
    ):
        # TODO: specify a matrix for storing output values
        # N datapoints, K classes
        N = x.shape[0]
        K = self.dataset.K
        y = np.zeros((N, K))

        # TODO: implement the feed-forward layer operations
        # 1. Specify a loop over all the datapoints
        # 2. Specify the input layer (2x1 matrix)
        # 3. For each hidden layer, perform the MLP operations
        #    - multiply weight matrix and output from previous layer
        #    - add bias vector
        #    - apply activation function
        # 4. Specify the final layer, with 'softmax' activation
        for i in range(N):
            # input column vector (D x 1)
            h = x[i, :][:, np.newaxis]

            # hidden layers
            for layer in range(self.hidden_layers):
                z = np.dot(self.W[layer], h) + self.b[layer]
                h = activation(z, self.activation)

            # output layer with softmax
            z_out = np.dot(self.W[-1], h) + self.b[-1]
            h_out = activation(z_out, 'softmax')

            # store as row in y
            y[i, :] = h_out[:, 0]

        return y

    # Measure performance of model
    def evaluate(self):
        print('Model performance:')

        # TODO: formulate the training loss and accuracy of the MLP
        # Assume the mean squared error loss
        # Hint: For calculating accuracy, use np.argmax to get predicted class
        yp_train = self.feedforward(self.dataset.x_train)
        train_loss = np.mean((yp_train - self.dataset.y_train_oh) ** 2)
        train_pred = np.argmax(yp_train, axis=1)
        train_acc = np.mean(train_pred == self.dataset.y_train)

        print("\tTrain loss:     %0.4f" % train_loss)
        print("\tTrain accuracy: %0.2f" % train_acc)

        # TODO: formulate the test loss and accuracy of the MLP
        yp_test = self.feedforward(self.dataset.x_test)
        test_loss = np.mean((yp_test - self.dataset.y_test_oh) ** 2)
        test_pred = np.argmax(yp_test, axis=1)
        test_acc = np.mean(test_pred == self.dataset.y_test)

        print("\tTest loss:      %0.4f" % test_loss)
        print("\tTest accuracy:  %0.2f" % test_acc)
