from data_handling import get_dataset
import sparky
import numpy as np
import tensorflow as tf
from skopt import gp_minimize, forest_minimize
import multiprocessing
from random import sample
import time
from tqdm import tqdm
import warnings

## quality of life stuff
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.set_printoptions(threshold=np.inf)

## GLOBAL VARS FOR CONVENIENCE
Model = sparky.sparsernet
batch_size = 1
mini_batch_size = 1
verbose_error = False

## get/repackage training dataset
training_dataset = get_dataset(
      '../data/next_day_wildfire_spread_train*',
      data_size=64,
      sample_size=64,
      batch_size=batch_size,
      num_in_channels=12,
      compression_type=None,
      clip_and_normalize=True,
      clip_and_rescale=False,
      random_crop=True,
      center_crop=False)
inputs, labels = next(iter(training_dataset))
firesdata = np.append(inputs.numpy(), labels.numpy(), axis = 3)

def objective_function(parameters):
    t0 = time.time()
    #progress_bar = tqdm(total=mini_batch_size, desc="calculating error")

    model = Model(parameters = parameters)
    losses = np.zeros(4) ## order is [pred_total, true_total, unweighted, weighted]

    mini_batch_ids = sample(range(batch_size), mini_batch_size)

    # make this stochastic (iterating through all 18k would be intractable and iterating through 
    # small batches will probably be info-less/encourage overfitting)
    for i in mini_batch_ids:

        pred_next_front = model.predict([firesdata[i, :, :, j] for j in range(12)])

        # calculate loss on predicted next front
        true_next_front = firesdata[i, :, :, -1]
        losses += sparky.loss(true_next_front, pred_next_front)

        #progress_bar.update(1)

    tf = time.time()
    pred_fire_size, true_fire_size, unweighted, loss = losses

    regularization = 100 * (np.sum((np.abs(np.array(parameters))) ** 2) / (np.sum((np.array((2, 2) * int(np.ceil(model.nparams / 2)))) ** 2) + 100))      ## 100 * % of maximal parameter values

    if verbose_error :
        print(f"error calculation summary (across {mini_batch_size} examples) :")
        print(f"  > loss averages : (pred/true fire size : {pred_fire_size / mini_batch_size} / {true_fire_size / mini_batch_size},  unweighted loss : {unweighted / mini_batch_size},  damage-weighted loss : {loss / mini_batch_size}")
        print(f"  > regularization penalty : {np.round((1/8) * regularization, 3)}")
        print(f"  > total weighted loss (objective function) : {np.round(((5 / 8) * loss) + ((3/8) * regularization), 3)}")
        print(f"  > elapsed time : {tf - t0} sec")

    loss = ((5 / 8) * loss) + ((3/8) * regularization)
    
    return loss

## create instance of model to make param counting easier
nparams = Model().nparams
print(f"nparams : {nparams}")

print("> testing manual-param setting/objective calculation outside of optimization environment...")
test_params = np.random.rand(nparams)
test_params = np.zeros(nparams)
test_obj = objective_function(test_params)

"""SKOPT IMPLEMENTATION(S)"""
param_bounds = [(-2.0, 2.0) for i in range(nparams)]
param_bounds[nparams - 4] = (-100, 10)               ## allow much lower global intercept for spread rate

n_calls, n_initial_points = 100, 50

##gp_minimize
print(f"> initializing a gp_minimize search over {nparams} variables")
opt = gp_minimize(objective_function, dimensions = param_bounds, n_calls = n_calls, n_initial_points = n_initial_points)

##forest_minimize
print(f"> initializing a forest_minimize search over {nparams} variables")
opt2 = forest_minimize(objective_function, dimensions = param_bounds, n_calls = n_calls, n_initial_points = n_initial_points)

print(f"GP OPT BEST : ")
for i in range(mini_batch_size):
    print(f"gp_min optimal model's performance on mini-batch example {i}")
    sparky.summarize_performance(Model(parameters = opt.x), firesdata[i, :, :, :])

print(f"FOREST OPT BEST : ")
for i in range(mini_batch_size):
    print(f"forest_min optimal model's performance on mini-batch example {i}")
    sparky.summarize_performance(Model(parameters = opt2.x), firesdata[i, :, :, :])

"""EVO ALGO IMPLEMENTATION""""""
param_bounds = np.array([[-10.0, 10.0] for i in range(nparams)]) ## FIXME: make more informed choice here
evo_algo = ga(function = objective_function, dimension = nparams, variable_type = 'real', variable_boundaries = param_bounds, function_timeout = 100)
## run evo algo
evo_algo.run()
## view best parameters
evo_algo.report
evo_algo.output_dict """