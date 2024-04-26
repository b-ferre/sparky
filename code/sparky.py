import numpy as np
from collections import deque
import tensorflow as tf
import tensorflow as tf

###############################################################################################################
##                                                HELPERS                                                    ##
###############################################################################################################

## helper which uses BFS to generate distance matrix from fire-front matrix
def dist_to_front(matrix):
    rows, cols = matrix.shape
    
    # Replace solid fronts with front borders
    borders = np.copy(matrix)
    padding = np.pad(matrix, 1, mode='constant', constant_values=0)
    neighbors = np.array([padding[i-1:i+2, j-1:j+2] for i in range(1, rows+1) for j in range(1, cols+1)])
    neighbors_sum = np.sum(neighbors, axis=(1, 2))
    borders[matrix == 1] = (neighbors_sum[matrix.ravel() == 1] != 9)
    
    # Use BFS to calculate shortest distance to a border
    distances = np.zeros_like(matrix, dtype=float)
    queue = deque()
    distances[borders == 1] = 0
    queue.extend(np.transpose(np.nonzero(borders == 1)))
    
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < rows) and (0 <= ny < cols) and ((distances[nx, ny] == 0) or (distances[nx, ny] > distances[x, y] + 1)):
                if borders[nx, ny] == 0:
                    distances[nx, ny] = distances[x, y] + 1
                    queue.append((nx, ny))
        for dx, dy in [(1, 1), (-1, -1), (-1, 1), (1, -1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < rows) and (0 <= ny < cols) and ((distances[nx, ny] == 0) or (distances[nx, ny] > distances[x, y] + np.sqrt(2))):
                if borders[nx, ny] == 0:
                    distances[nx, ny] = distances[x, y] + np.sqrt(2)
                    queue.append((nx, ny))
    
    # Replace all internal cell distances with their negation
    distances[np.logical_and(matrix == 1, np.all(borders[1:-1, 1:-1] == 0, axis=(0, 1)))] *= -1
    distances = np.round(distances, 3)
    
    return distances

## helper that, given a numerical matrix with entries representing function values,
## estimates the gradient of said function at every point
def estimate_gradient(matrix):
    matrix = np.array(matrix, dtype=float)
    rows, cols = matrix.shape

    # Compute gradient_x
    gradient_x = np.zeros_like(matrix)
    gradient_x[:, :-1] = np.diff(matrix, axis=1)
    gradient_x[:, -1] = matrix[:, -1] - matrix[:, -2]

    # Compute gradient_y
    gradient_y = np.zeros_like(matrix)
    gradient_y[:-1, :] = np.diff(matrix, axis=0)
    gradient_y[-1, :] = matrix[-1, :] - matrix[-2, :]

    return gradient_x, gradient_y

## prints front matrix with one row per line and all entries evenly spaced
def print_matrix(matrix):
    matrix = np.round(matrix, 3)
    np.set_printoptions(linewidth=np.inf)  # Disable line-wrapping
    # Determine the maximum width needed for each element
    max_width = max(len(str(max(row))) for row in matrix)

    # Print each row with even spacing
    for row in matrix:
        formatted_row = ' '.join(f'{num:{max_width}}' for num in row)
        print(formatted_row)

## code that mimics SPARK's level-set method FIXME: be more thorough w this if time (see notes in function)
def spark_iter(current_front, spread_rates):
    ## - current_front and spread_rates must be the same dimension
    assert(current_front.shape == spread_rates.shape)
    ## - current front is a binary raster where (current_front[i, j] = 1) ==> there is fire in grid cell (i, j)
    ## - spread rates is a numerical raster where each cell contains the spread rate of that cell (which is >= 0,
    ## and should only be 0 if the cell contains entirely unburnable material like water)

    ## generate new front
    new_front = np.zeros(current_front.shape)

    ## step one: compute phi for each cell
    phi = dist_to_front(current_front)

    ## FIXME: this is definitely not the typical way of implementing a level-set method like this one, 
    ## and by approximating gradients/discretizing the distance function, we are losing a ton of info/
    ## flexibility. Unfortunately the original paper provides little to no information on how they
    ## implemented the level-set method, but we need to figure that out and then fix this function
    ## accordingly.
    
    ## estimate gradient at each cell
    x_grad, y_grad = estimate_gradient(phi)
    abs_grad_phi = np.sqrt((x_grad ** 2) + (y_grad ** 2))

    ## apply level-set propagation equation to each cell by computing element-wise product of spread and grad_phi
    delta_phi = -1 * np.multiply(abs_grad_phi, spread_rates)
    new_phi = phi + delta_phi

    ## update front to reflect new distances. dist to front is considered to be from the center of the grid so if 
    ## dist < 0.5, there is fire inside the grid cell, so we consider it "on fire"
    new_front[new_phi <= 0.5] = 1

    return(new_front)

## TODO: use a more apt loss than squared error lol
def loss(true_next_front, pred_next_front):
    loss = 0

    diff = true_next_front - pred_next_front
    diff = np.matrix(diff).flatten()
    loss = np.sum(np.abs(diff))

    ## extra error term that heavily penalizes missing fire size (worse for underestimating)
    pred_fire_size = np.sum(pred_next_front)
    true_fire_size = np.sum(true_next_front[true_next_front >= 0])
    loss += (10 * (max(0, true_fire_size - pred_fire_size))) + (2 * (max(0, pred_fire_size - true_fire_size)))
    return loss

def test_helpers():
    test_front = np.zeros((20, 20))
    test_front[10, 10] = 1
    print("")
    print_matrix(test_front)
    print("... \n dist:")
    print_matrix(dist_to_front(test_front))
    gradient_x, gradient_y = estimate_gradient(dist_to_front(test_front))
    print("... \n grad x:")
    print_matrix(gradient_x)
    print("... \n grad y:")
    print_matrix(gradient_y)
    print("... \n |nabla|:")
    print_matrix(np.sqrt((gradient_x ** 2) + ((gradient_y) ** 2)))

###############################################################################################################
##                                              MODEL CODE                                                   ##
###############################################################################################################


## TODO: replace boiler plate NN architecture with application specific one
class sparknet():

    def __init__(self, layer_sizes = [49152, 4096], 
                    activations = ['softmax'],
                    parameters = None):

        print(f"sparky initializing... {layer_sizes}")

        self.layer_sizes = layer_sizes
        self.activations = activations

        # initialize weights randomly
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i-1]) for i in range(1, len(layer_sizes))]
        self.biases = [np.random.randn(layer_sizes[i], 1) for i in range(1, len(layer_sizes))]
        print(len(self.weights))
        print(len(self.weights))

        # define useful constants
        self.nweights = np.sum(list(map(np.product, [w.shape for w in self.weights])))
        self.nbiases = np.sum(layer_sizes[1:])

        # set parameters manually if they are passed
        if ((parameters is not None) and (len(parameters) == self.nweights + self.nbiases)):
            print(f"manually setting parameters. length of param vector is {len(parameters)}")
            print(len(self.weights))
            print(len(self.biases))
            j = 0
            for i in range(len(self.weights)):
                print(f"updating weights on layer {i}")
                intermediate = parameters[j:(j + (layer_sizes[i+1] * layer_sizes[i]))]
                print(len(intermediate))
                print(layer_sizes[i])
                print(layer_sizes[i+1])
                self.weights[i] = np.array(parameters[j:(j + (layer_sizes[i+1] * layer_sizes[i]))]).reshape(layer_sizes[i+1], layer_sizes[i])
                j +=  (layer_sizes[i + 1] * layer_sizes[i])
            for i in range(len(self.biases)):
                print(f"updating biases on layer {i}")
                self.biases[i] = np.array(parameters[j:(j + layer_sizes[i + 1])])
                j += layer_sizes[i + 1]

        print(f"spark init complete; {len(self.weights)}, {len(self.biases)}")
    
    def forward(self, X):
        A = X
        print("predicting...")
        print(len(self.weights))
        for i in range(len(self.weights)):
            print(f"forward propagating X (shape : {A.shape} through layer {i}...")
            print(f"  > left-multiplying by W (shape : {self.weights[i].shape})")
            Z = np.dot(self.weights[i], A) + self.biases[i]
            print(f"  > applying activation {self.activations[i]} to Z = WX (shape  : {Z.shape})")
            A = self.activation_function(Z, self.activations[i])
            print(f"final shape of h(WX) is {A.shape}")
        print(f"returning predictions Y_hat (shape : {A.shape})")
        return A
    
    def activation_function(self, Z, activation):
        if activation == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        elif activation == 'relu':
            return np.maximum(0, Z)
        elif activation == 'tanh':
            return np.tanh(Z)
        elif activation == 'softmax':
            expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            return expZ / np.sum(expZ, axis=0, keepdims=True)
        else:
            raise ValueError("Activation function not supported.")
    
    def predict(self, X):
        return self.forward(X)


## less parameters => more easily optimized
## 2,000,000+ params (as in even basic 2 layer fully-connected sparsenet) is intractable

class sparsenet():

    ### constants



    def __init__(self, layer_sizes = [49152, 256, 4096], 
                    activations = ['sigmoid', 'softmax'],
                    parameters = None):

        print(f"sparky initializing... {layer_sizes}")

        self.layer_sizes = layer_sizes
        self.activations = activations

        # initialize weights randomly
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i-1]) for i in range(1, len(layer_sizes))]
        self.biases = [np.random.randn(layer_sizes[i], 1) for i in range(1, len(layer_sizes))]
        print(len(self.weights))
        print(len(self.weights))

        # define useful constants
        self.nweights = np.sum(list(map(np.product, [w.shape for w in self.weights])))
        self.nbiases = np.sum(layer_sizes[1:])

        # set parameters manually if they are passed
        if ((parameters is not None) and (len(parameters) == self.nweights + self.nbiases)):
            print(f"manually setting parameters. length of param vector is {len(parameters)}")
            print(len(self.weights))
            print(len(self.biases))
            j = 0
            for i in range(len(self.weights)):
                print(f"updating weights on layer {i}")
                intermediate = parameters[j:(j + (layer_sizes[i+1] * layer_sizes[i]))]
                print(len(intermediate))
                print(layer_sizes[i])
                print(layer_sizes[i+1])
                self.weights[i] = np.array(parameters[j:(j + (layer_sizes[i+1] * layer_sizes[i]))]).reshape(layer_sizes[i+1], layer_sizes[i])
                j +=  (layer_sizes[i + 1] * layer_sizes[i])
            for i in range(len(self.biases)):
                print(f"updating biases on layer {i}")
                self.biases[i] = np.array(parameters[j:(j + layer_sizes[i + 1])])
                j += layer_sizes[i + 1]

        print(f"spark init complete; {len(self.weights)}, {len(self.biases)}")
    
    def forward(self, X):
        A = X
        print("predicting...")
        print(len(self.weights))
        for i in range(len(self.weights)):
            print(f"forward propagating X (shape : {A.shape} through layer {i}...")
            print(f"  > left-multiplying by W (shape : {self.weights[i].shape})")
            Z = np.dot(self.weights[i], A) + self.biases[i]
            print(f"  > applying activation {self.activations[i]} to Z = WX (shape  : {Z.shape})")
            A = self.activation_function(Z, self.activations[i])
            print(f"final shape of h(WX) is {A.shape}")
        print(f"returning predictions Y_hat (shape : {A.shape})")
        return A
    
    def activation_function(self, Z, activation):
        if activation == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        elif activation == 'relu':
            return np.maximum(0, Z)
        elif activation == 'tanh':
            return np.tanh(Z)
        elif activation == 'softmax':
            expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            return expZ / np.sum(expZ, axis=0, keepdims=True)
        else:
            raise ValueError("Activation function not supported.")
    
    def predict(self, X):
        return self.forward(X)