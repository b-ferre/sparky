from data_handling import get_dataset
import sparky
import numpy as np
import tensorflow as tf
from skopt import gp_minimize
from geneticalgorithm import geneticalgorithm as ga
import multiprocessing

## GLOBAL VARS FOR CONVENIENCE
batch_size = 10  ## FIXME: up this, add stochasticity, and/or parallelize error computation on slurm

## get/repackage training dataset
training_dataset = get_dataset(
      '../data/next_day_wildfire_spread_train*',
      data_size=64,
      sample_size=64,
      batch_size=batch_size,
      num_in_channels=12,
      compression_type=None,
      clip_and_normalize=False,
      clip_and_rescale=False,
      random_crop=True,
      center_crop=False)
inputs, labels = next(iter(training_dataset))
firesdata = np.append(inputs.numpy(), labels.numpy(), axis = 3)

def objective_function(parameters):
    model = sparky.sparsenet(parameters = parameters)
    loss = 0
    # make this stochastic (iterating through all 18k would be intractable and iterating through 
    # small batches will probably be info-less/encourage overfitting)
    for i in range(batch_size):

        # predict spread rates using sparknet, get current fire front
        spread_rates = model.predict([firesdata[i, :, :, j] for j in range(12)])
        current_front = firesdata[i, :, :, 12]

        # use predcited spread rates and current front to predict next front
        pred_next_front = sparky.spark_iter(current_front, spread_rates)

        # calculate loss on predicted next front
        true_next_front = firesdata[i, :, :, -1]
        loss += sparky.loss(true_next_front, pred_next_front)
    print(f"loss : {loss}")    
    return loss

## create instance of model to make param counting easier
model = sparky.sparsenet()
nparams = model.nparams

print("> testing manual-param setting/objective calculation outside of optimization environment...")
test_params = np.random.rand(nparams)
test_obj = objective_function(test_params)

### SKOPT IMPLEMENTATION
print(f"> initializing a gp_minimize search over {nparams} variables")
param_bounds = [(-10.0, 10.0) for i in range(nparams)] ## FIXME: make more informed choice here
opt = gp_minimize(objective_function, dimensions = param_bounds, n_calls = 100, n_initial_points = 10)
print(opt)

"""
### EVO ALGO IMPLEMENTATION
param_bounds = np.array([[-10.0, 10.0] for i in range(nparams)]) ## FIXME: make more informed choice here
evo_algo = ga(function = objective_function, dimension = nparams, variable_type = 'real', variable_boundaries = param_bounds, function_timeout = 100)
## run evo algo
evo_algo.run()
## view best parameters
evo_algo.report
evo_algo.output_dict """