import numpy as np
from collections import deque
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

## code that mimics SPARK's level-set method
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

    diff = true_next_front[true_next_front != 0] - pred_next_front[true_next_front != 0]
    diff = np.matrix(diff).flatten()
    loss = np.sum(np.abs(diff))

    ## extra error term that heavily penalizes missing fire size (worse for underestimating)
    pred_fire_size = np.sum(pred_next_front)
    true_fire_size = np.sum(true_next_front[true_next_front >= 0])
    loss += (10 * (max(0, true_fire_size - pred_fire_size))) + (2 * (max(0, pred_fire_size - true_fire_size)))

    return np.array([pred_fire_size, true_fire_size, np.sum(np.abs(diff)), loss])

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


## incredibly boiler-plate fully connected neural network with plug and play parameters for external optimization 
## could be great if we had infinite compute but the black box nature of our loss function (in terms of model params)
## means that this is infeasible
class sparknet():

    ## class helper
    def vprint(self, x):
        if self.verbose == True:
            print(x)

    def __init__(self, layer_sizes = [49152, 4096], 
                    activations = ['softmax'],
                    parameters = None, verbose = True):

        self.verbose = verbose
        self.vprint(f"> sparky initializing... {layer_sizes}")

        self.layer_sizes = layer_sizes
        self.activations = activations

        # initialize weights randomly
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i-1]) for i in range(1, len(layer_sizes))]
        self.biases = [np.random.randn(layer_sizes[i], 1) for i in range(1, len(layer_sizes))]

        # define useful constants
        self.nweights = np.sum(list(map(np.prod, [w.shape for w in self.weights])))
        self.nbiases = np.sum(layer_sizes[1:])

        # set parameters manually if they are passed
        if ((parameters is not None) and (len(parameters) == self.nweights + self.nbiases)):
            self.vprint(f"    > manually setting parameters. length of param vector is {len(parameters)}")
            j = 0
            for i in range(len(self.weights)):
                intermediate = parameters[j:(j + (layer_sizes[i+1] * layer_sizes[i]))]
                self.weights[i] = np.array(parameters[j:(j + (layer_sizes[i+1] * layer_sizes[i]))]).reshape(layer_sizes[i+1], layer_sizes[i])
                j +=  (layer_sizes[i + 1] * layer_sizes[i])
            for i in range(len(self.biases)):
                self.biases[i] = np.array(parameters[j:(j + layer_sizes[i + 1])])
                j += layer_sizes[i + 1]

        self.vprint(f"> sparky init complete!")
    
    def forward(self, X):
        A = X
        self.vprint("> predicting...")
        for i in range(len(self.weights)):
            Z = np.dot(self.weights[i], A) + self.biases[i]
            A = self.activation_function(Z, self.activations[i])
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
## with default hyper-parameters, this model has just shy of 100,000 params
## convolution_size must be odd
class sparsenet():

    ## parameters is a 1d vector of length sparsenet.nparams
    def format_params(self, parameters):
        if ((parameters is None) or (len(parameters) != self.nparams)):
            return None
        else :
            print(parameters)
            parameters = np.array(parameters)
            params = {}
            params['local_NLL_regression_weights'] = parameters[0:36].reshape(3, 12)

            conv = []
            for q in range(12):
                conv.append(parameters[(q * (self.convolution_size ** 2)) : ((q + 1) * (self.convolution_size ** 2))].reshape(self.convolution_size, self.convolution_size))
            params['convolutions'] = conv

            j = 36 + (12 * (self.convolution_size ** 2))
            params['convolution_regression_weights'] = parameters[j: j + 36].reshape(3, 12)
            j += 36

            k = j + ((64 ** 2) * 12 * self.global_nodes)
            params['global_net_weights'] = parameters[j : k].reshape(self.global_nodes, (64 * 64 * 12))
            params['global_net_biases'] = parameters[k:(k + self.global_nodes)]
            params['global_regression_weights'] = parameters[k:(k + self.global_nodes)]
            
            params['component_weights'] = parameters[(k + self.global_nodes): ]
            return params

    def perform_convolution(self, matrix, kernel):
        ## TODO: check this is right; this is GPT code lol
        m, n = matrix.shape
        k, _ = kernel.shape
        result = np.zeros((m - k + 1, n - k + 1))

        for i in range(m - k + 1):
            for j in range(n - k + 1):
                window = matrix[i:i + k, j:j + k]
                result[i, j] = np.sum(window * kernel)

        return result

    def __init__(self, parameters = None,
                global_nodes = 1,
                convolution_size = 3,
                activations = ['sigmoid', 'sigmoid', 'sigmoid']):

        print(f"sparse-ky initializing...")

        self.global_nodes = global_nodes
        self.activations = activations
        self.convolution_size = convolution_size
        self.nparams = (36) + ((12 * (convolution_size ** 2)) + 36) + ((global_nodes * (64 * 64 * 12)) + (2 * global_nodes)) + (3)      ## see architecture building below for explanation
        self.params = self.format_params(parameters)

        if self.params == None:
            self.params = {}
            print("    > no parameters provided. initializing all weights randomly...")
            self.params['local_NLL_regression_weights'] = np.random.randn(3, 12) ## row one is weights, row two is biases, row three is post-activation regression

            self.params['convolutions'] = [np.random.randn(convolution_size, convolution_size) for i in range(12)]
            self.params['convolution_regression_weights'] = np.random.randn(3, 12) ## row one is weights, row two is biases, row three is post-activation regression

            self.params['global_net_weights'] = np.random.randn(global_nodes, (64 * 64 * 12))
            self.params['global_net_biases'] = np.random.randn(global_nodes)
            self.params['global_regression_weights'] = np.random.randn(global_nodes)

            self.params['component_weights'] = np.random.randn(3)

        print(f"spark init complete!")
    
    ## firedata needs to be a list of 12 64 x 64 matricies
    def forward(self, firedata):
        pred = np.zeros((64, 64))

        ## add weighted fully local regression component to predictions
        local_reg_activations= [self.activation_function(np.array(((self.params['local_NLL_regression_weights'][0, i] * firedata[i]) + self.params['local_NLL_regression_weights'][1, i])), self.activations[0]) for i in range(12)]
        local_reg_component = np.sum(np.array([(self.params['local_NLL_regression_weights'][2, i] * local_reg_activations[i]) for i in range(12)]), axis = 0)
        pred += local_reg_component * self.params['component_weights'][0] ## adds different values everywhere

        ## apply convolutional component
        to_conv = [np.pad(firedata[i], (int((self.convolution_size - 1) / 2),), 'constant', constant_values = 0) for i in range(12)]
        convd = [self.perform_convolution(to_conv[i], self.params['convolutions'][i]) for i in range(12)]
        conv_reg_activations= [self.activation_function(np.array(((self.params['convolution_regression_weights'][0, i] * convd[i]) + self.params['convolution_regression_weights'][1, i])), self.activations[1]) for i in range(12)]
        conv_reg_component = np.sum(np.array([(self.params['convolution_regression_weights'][2, i] * conv_reg_activations[i]) for i in range(12)]), axis = 0)
        pred += conv_reg_component * self.params['component_weights'][1] ## adds different values everywhere
        
        ## apply (shallow) global neural network component
        vectorized = np.stack(firedata, axis = -1).reshape(-1, )
        nn_activations = self.activation_function(np.dot(self.params['global_net_weights'], vectorized) + self.params['global_net_biases'], self.activations[2])
        nn_component = np.dot(nn_activations, self.params['global_regression_weights'])
        pred += nn_component * self.params['component_weights'][2]  ## this is adding a global scalar based on overall conditions

        return pred

    def activation_function(self, Z, activation):
        if activation == 'sigmoid':
            try:
                return 1 / (1 + np.exp(-Z))
            except RuntimeWarning:
                print("got one")
                print(-Z)
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


## how many parameters do we *reaally* need? is a global NN component overkill if we can't reasonably
## have a high-dimensional final layer, have to keep the model shallow, and are forced to have a tiny bottleneck?
## this is, as its name suggests, an even sparser sparsenet which just uses regression on global means
## (and a couple other global statistics) for its global component instead of a neural network-ish approach

## convolution_size must be odd
class sparsernet():

    ## class helper
    def vprint(self, x):
        if self.verbose == True:
            print(x)

    ## parameters is a 1d vector of length sparsenet.nparams
    def format_params(self, parameters):
        if ((parameters is None) or (len(parameters) != self.nparams)):
            return None
        else :
            parameters = np.array(parameters)
            params = {}
            params['local_NLL_regression_weights'] = parameters[0:36].reshape(3, 12)

            conv = []
            for q in range(12):
                conv.append(parameters[(q * (self.convolution_size ** 2)) : ((q + 1) * (self.convolution_size ** 2))].reshape(self.convolution_size, self.convolution_size))
            params['convolutions'] = conv

            j = 36 + (12 * (self.convolution_size ** 2))
            params['convolution_regression_weights'] = parameters[j: j + 36].reshape(3, 12)
            j += 36

            k = j + 36
            params['global_stat_weights'] = parameters[j : k]
            params['global_bias'] = parameters[k : k + 1]
            
            params['component_weights'] = parameters[k + 1: ]
            return params

    def perform_convolution(self, matrix, kernel):
        ## TODO: check this is right; this is GPT code lol
        m, n = matrix.shape
        k, _ = kernel.shape
        result = np.zeros((m - k + 1, n - k + 1))

        for i in range(m - k + 1):
            for j in range(n - k + 1):
                window = matrix[i:i + k, j:j + k]
                result[i, j] = np.sum(window * kernel)

        return result

    def __init__(self, verbose = False, parameters = None,
                convolution_size = 11,
                activations = ['sigmoid', 'sigmoid']):

        self.verbose = verbose
        self.vprint(f"sparse-ky initializing...")

        self.activations = activations
        self.convolution_size = convolution_size
        self.nparams = (36) + ((12 * (convolution_size ** 2)) + 36) + (36 + 1) + (3)      ## see architecture building below for explanation
        self.params = self.format_params(parameters)

        if self.params == None:
            self.params = {}
            self.vprint("    > no parameters provided. initializing all weights randomly...")
            self.params['local_NLL_regression_weights'] = np.random.randn(3, 12) ## row one is weights, row two is biases, row three is post-activation regression

            self.params['convolutions'] = [np.random.randn(convolution_size, convolution_size) for i in range(12)]
            self.params['convolution_regression_weights'] = np.random.randn(3, 12) ## row one is weights, row two is biases, row three is post-activation regression

            self.params['global_stat_weights'] = np.random.randn(36)
            self.params['global_bias'] = np.random.randn(1)

            self.params['component_weights'] = np.random.randn(3)

        self.vprint(f"spark init complete!")
    
    ## firedata needs to be a list of 12 64 x 64 matricies
    def forward(self, firedata):
        pred = np.zeros((64, 64))

        ## add weighted fully local regression component to predictions
        local_reg_activations= [self.activation_function(np.array(((self.params['local_NLL_regression_weights'][0, i] * firedata[i]) + self.params['local_NLL_regression_weights'][1, i])), self.activations[0]) for i in range(12)]
        local_reg_component = np.sum(np.array([(self.params['local_NLL_regression_weights'][2, i] * local_reg_activations[i]) for i in range(12)]), axis = 0)
        pred += local_reg_component * self.params['component_weights'][0] ## adds different values everywhere

        ## apply convolutional component
        to_conv = [np.pad(firedata[i], (int((self.convolution_size - 1) / 2),), 'constant', constant_values = 0) for i in range(12)]
        convd = [self.perform_convolution(to_conv[i], self.params['convolutions'][i]) for i in range(12)]
        conv_reg_activations= [self.activation_function(np.array(((self.params['convolution_regression_weights'][0, i] * convd[i]) + self.params['convolution_regression_weights'][1, i])), self.activations[1]) for i in range(12)]
        conv_reg_component = np.sum(np.array([(self.params['convolution_regression_weights'][2, i] * conv_reg_activations[i]) for i in range(12)]), axis = 0)
        pred += conv_reg_component * self.params['component_weights'][1] ## adds different values everywhere
        
        ## apply (shallow) global neural network component
        means = [np.mean(feature) for feature in firedata]
        medians = [np.median(feature) for feature in firedata]
        sds = [np.std(feature) for feature in firedata]
        global_stats = np.stack([means, medians, sds]).reshape(-1, )
        global_component = np.dot(global_stats, self.params['global_stat_weights']) + self.params['global_bias']
        pred += global_component * self.params['component_weights'][2]

        return pred

    def activation_function(self, Z, activation):
        if activation == 'sigmoid':
            try:
                return 1 / (1 + np.exp(-Z))
            except RuntimeWarning:
                print("got one")
                print(-Z)
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

        