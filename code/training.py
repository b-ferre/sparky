from geneticalgorithm import geneticalgorithm as ga
from data_handling import get_dataset
from sparky import spark_iter, print_matrix
import numpy as np
import tensorflow as tf
import sklearn
from skopt import gp_minimize
from sklearn.neural_network import MLPClassifier
import multiprocessing

## GLOBAL VARS FOR CONVENIENCE
sample_size = 64
batch_size = 5  ## FIXME: up this, add stochasticity, and/or parallelize error computation on slurm;
## (low for now so I can run proof of concept of this algorithm locally and at least overfit a NN to like 5 wildfires)

## get training, validation datasets
training_dataset = get_dataset(
      '../data/next_day_wildfire_spread_train*',
      data_size=64,
      sample_size=sample_size,
      batch_size=batch_size,
      num_in_channels=12,
      compression_type=None,
      clip_and_normalize=False,
      clip_and_rescale=False,
      random_crop=True,
      center_crop=False)

#TODO: use more examples, not just first 10; also make this stochastic (iterating through all 18k would be intractable)
inputs, labels = next(iter(training_dataset))
print(inputs.numpy().shape)

firesdata = np.append(inputs.numpy(), labels.numpy(), axis = 3)
print(firesdata.shape)

# Define the neural network architecture
## TODO: replace generic NN architecture with an application specific one
class NeuralNetwork:
    def __init__(self, layer_sizes = [768, 512, 256, sample_size], 
                    activations = ['sigmoid', 'sigmoid', 'sigmoid', 'relu', 'softmax'],
                    weights = None, biases = None):
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i-1]) for i in range(1, len(layer_sizes))]
        self.biases = [np.random.randn(layer_sizes[i], 1) for i in range(1, len(layer_sizes))]
        
        if ((weights is not None) and (biases is not None)):
            self.weights = weights
            self.biases = biases
    
    def forward(self, X):
        A = X
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

## objective function with embedded model
## TODO: search over/ decide on better NN architecture/hyper-params

def objective_function(parameters):

    def single_fire_loss(info):
        firedata, parameters = info

        model = NeuralNetwork(weights = parameters[0:3], biases = parameters[3:])

        spread_rates = model.predict(firedata[:, :, 0:12].numpy().reshape(-1, sample_size)).reshape(sample_size, sample_size) ##FIXME: did this adhoc. make sure this works

        current_front = firedata[:, :, 12]

        pred_next_front = spark_iter(current_front, spread_rates)

        true_next_front = firedata[:, :, -1]

        ## TODO: use a more apt loss than squared error lol
        diff = true_next_front - pred_next_front
        diff = np.matrix(diff).flatten()
        loss = np.sum(np.abs(diff))

        ## extra error term that heavily penalizes missing fire size (worse for underestimating)
        pred_fire_size = np.sum(pred_next_front)
        true_fire_size = np.sum(true_next_front[true_next_front >= 0])
        loss += (10 * (max(0, true_fire_size - pred_fire_size))) + (2 * (max(0, pred_fire_size - true_fire_size)))

        return loss

    pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())

    losses = pool.map(single_fire_loss, [firesdata, params])
    loss = np.sum(losses)
    print(f"total loss : {loss}")
    

## create instance of model to make param counting easier
model = NeuralNetwork()
print(model.layer_sizes)
nweights = np.sum(np.array([np.sum(np.array([len(item_) for item_ in item])) for item in model.weights]))
nbiases = nweights = np.sum(np.array([np.sum(np.array([len(item_) for item_ in item])) for item in model.weights]))
nparams = nweights + nbiases
print(f"nweights : {nweights}, nbiases: {nbiases}")
param_bounds = [[(-100.0, 100.0, 'uniform')] * nparams]

### SKOPT IMPLEMENTATION
opt = gp_minimize(objective_function, dimensions = param_bounds, n_calls = 1000, n_initial_points = 100)
print(opt)

### EVO ALGO IMPLEMENTATION
""" evo_algo = ga(function = objective_function, dimension = nparams, variable_type = 'real', variable_boundaries = param_bounds, function_timeout = 100)

## run evo algo
evo_algo.run()

## view best parameters
evo_algo.report
evo_algo.output_dict """