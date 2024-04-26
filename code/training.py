from data_handling import get_dataset
import sparky
import numpy as np
import tensorflow as tf
from skopt import gp_minimize
import multiprocessing
from random import sample
import time
from tqdm import tqdm

## quality of life add-in because of overflow in exp calculation
import warnings
# Silencing only RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

## GLOBAL VARS FOR CONVENIENCE
batch_size = 3000  ## FIXME: up this, add stochasticity, and/or parallelize error computation on slurm
mini_batch_size = 100

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

loss_record = []

def objective_function(parameters):

    t0 = time.time()
    progress_bar = tqdm(total=mini_batch_size, desc="calculating error")

    model = sparky.sparsernet(parameters = parameters)
    losses = np.zeros(4) ## order is [pred_total, true_total, unweighted, weighted]

    mini_batch_ids = sample(range(batch_size), mini_batch_size)

    # make this stochastic (iterating through all 18k would be intractable and iterating through 
    # small batches will probably be info-less/encourage overfitting)
    for i in mini_batch_ids:

        # predict spread rates using sparknet, get current fire front
        spread_rates = model.predict([firesdata[i, :, :, j] for j in range(12)])
        current_front = firesdata[i, :, :, 12]

        # use predcited spread rates and current front to predict next front
        pred_next_front = sparky.spark_iter(current_front, spread_rates)

        # calculate loss on predicted next front
        true_next_front = firesdata[i, :, :, -1]
        losses += sparky.loss(true_next_front, pred_next_front)

        progress_bar.update(1)

    tf = time.time()
    pred_fire_size, true_fire_size, unweighted, loss = losses
    loss_record.append(loss)

    print(f"error calculation summary (across {mini_batch_size} examples) :")
    print(f"  > loss averages : (pred/true fire size : {pred_fire_size / mini_batch_size} / {true_fire_size / mini_batch_size},  unweighted loss : {unweighted / mini_batch_size},  weighted loss : {loss / mini_batch_size}")
    print(f"  > total weighted loss (objective function) : {loss}")
    print(f"  > elapsed time : {tf - t0} sec")
    
    return loss

## create instance of model to make param counting easier
model = sparky.sparsernet()
nparams = model.nparams
print(f"nparams : {nparams}")

print("> testing manual-param setting/objective calculation outside of optimization environment...")
test_params = np.random.rand(nparams)
test_obj = objective_function(test_params)

### SKOPT IMPLEMENTATION
print(f"> initializing a gp_minimize search over {nparams} variables")
param_bounds = [(-10.0, 10.0) for i in range(nparams)] ## FIXME: make more informed choice here
opt = gp_minimize(objective_function, dimensions = param_bounds, n_calls = 500, n_initial_points = 30)
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