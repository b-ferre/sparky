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
batch_size = 1000
mini_batch_size = 1
verbose_error = False
optimizer = forest_minimize
n_calls, n_initial_points = 310, 300

progress_bar = tqdm(total=n_calls, desc="calculating error")

## get/repackage training dataset
training_dataset = get_dataset(
      './data/next_day_wildfire_spread_train*',
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

## move this back inside objective function to make search stochastic
mini_batch_ids = sample(range(batch_size), mini_batch_size)
print("prev_front : ")
sparky.print_matrix(firesdata[mini_batch_ids[0], :, :, 11])
print(""), print("")
print("true_next_front : ")
sparky.print_matrix(firesdata[mini_batch_ids[0], :, :, 12])
print(""), print("")
lol = input("train on this fire?")
while(lol == "no"):
    mini_batch_ids = sample(range(batch_size), mini_batch_size)
    sparky.print_matrix(firesdata[mini_batch_ids[0], :, :, 11])
    print(""), print("")
    sparky.print_matrix(firesdata[mini_batch_ids[0], :, :, 12])
    lol = input()

def objective_function(parameters):
    global progress_bar
    progress_bar.update(1)

    t0 = time.time()
    #progress_bar = tqdm(total=mini_batch_size, desc="calculating error")

    model = Model(parameters = parameters)
    losses = np.zeros(4) ## order is [pred_total, true_total, unweighted, weighted]

    # make this stochastic (iterating through all 18k would be intractable and iterating through 
    # small batches will probably be info-less/encourage overfitting)
    for i in mini_batch_ids:

        pred_next_front = model.predict([firesdata[i, :, :, j] for j in range(12)])

        # calculate loss on predicted next front
        true_next_front = firesdata[i, :, :, -1]
        losses += sparky.loss(true_next_front, pred_next_front)

    tf = time.time()
    pred_fire_size, true_fire_size, diff, loss = losses

    if verbose_error :
        print("..."), print("")
        print(f"error calculation summary (across {mini_batch_size} examples) :")
        print(f"  > weighted ADI (obj function) : {loss}")
        print(f"  > elapsed time : {tf - t0} sec")
        print("")

        if loss < 5:
            time.sleep(3)
            sparky.summarize_performance(model, firesdata[0, :, :, :])
            time.sleep(3)
    
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

"""
##gp_minimize
print(f"> initializing a gp_minimize search over {nparams} variables with {n_calls} total calls")
call_no = 1
opt = gp_minimize(objective_function, dimensions = param_bounds, n_calls = n_calls, n_initial_points = n_initial_points)"""


##forest_minimize
print(f"> initializing a forest_minimize search over {nparams} variables with {n_calls} total calls")
opt = optimizer(objective_function, dimensions = param_bounds, n_calls = n_calls, n_initial_points = n_initial_points)

"""
print(f"GP OPT BEST : ")
for i in mini_batch_ids:
    print(f"gp_min optimal model's performance on mini-batch example {i}")
    sparky.summarize_performance(Model(parameters = opt.x), firesdata[i, :, :, :]) """

print(f"FOREST OPT BEST : ")
for i in mini_batch_ids:
    print(f"forest_min optimal model's performance on mini-batch example {i}")
    sparky.summarize_performance(Model(parameters = opt.x), firesdata[i, :, :, :])
