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
batch_size = 12
mini_batch_size = 1
verbose_error = False
optimizer = gp_minimize
n_calls, n_initial_points = 10, 0
call_no, progress_bar = 0, None

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

"""
inputs, labels = next(iter(training_dataset))
firesdata = np.concatenate((inputs.numpy(), labels.numpy()), axis = 3)
 mini_batch_ids = sample(range(batch_size), mini_batch_size)"""

## right now this code block is set up to get the example with index 5 in the training dataset
## this is all ad-hoc experimentation; clean up later
counter = tf.data.Dataset.counter()
training_dataset = tf.data.Dataset.zip((training_dataset, counter))

#FIXME: this is currently set so I can do a ton of pre-compute for one fire to prove overfitting. replace later.
firesdata = None
mini_batch_ids = [0]
xstash, ystash = list(np.load("xstash.npy"))[0:500], list(np.load("ystash.npy"))[0:500]
for item, count in training_dataset:
    if count.numpy() == 39:
        firesdata = np.concatenate([item[i] for i in range(2)], axis = 3)
    # check = np.concatenate([item[i] for i in range(2)], axis = 3)
    # print(f"fire at abs_index {count}: ")
    # time.sleep(0.5)
    # sparky.print_matrix(check[0, :, :, 11])
    # print(""), print("")
    # sparky.print_matrix(check[0, :, :, 12])
    # time.sleep(5)

### SOME USEFUL ABS_INDEXES:
"""
- 10 : fire moves laterally and does not change size much. can we model this?
- 24, 25, 35 : small fires that change shape/spread slightly
- 26 : single cell fire which gets extinguished
- 33 : good medium size fire with some corruption
- 38, 41 : medium+, complex
- 39 : medium, easy
- 42 : large, complex
"""

print("training on this fire : ")
time.sleep(0.5)
sparky.print_matrix(firesdata[0, :, :, 11])
print(""), print("")
sparky.print_matrix(firesdata[0, :, :, 12])

"""
## code for manually choosing a fire to train on
lol, abs_index = "no", 0
while(lol != "yes"):
    for item, count in training_dataset:
        if count.numpy() == abs_index:
            firesdata = np.concatenate([item[i] for i in range(2)], axis = 3)
    sparky.print_matrix(firesdata[0, :, :, 11])
    print(""), print("")
    sparky.print_matrix(firesdata[0, :, :, 12])
    lol = input(f"train on this fire (abs_index : {abs_index})?\n")
    abs_index += 1
    if (lol == "q"):
        del training_dataset
        quit()
"""
def objective_function(parameters):
    global xstashh, ystashh
    global call_no
    global progress_bar
    if ((call_no == n_initial_points)):
        progress_bar = tqdm(total=n_calls - n_initial_points, desc="running gaussian process estimator...")
    elif ((call_no == 0)):
        progress_bar = tqdm(total=n_initial_points, desc="initializing gaussian process estimator...")
    else:
        progress_bar.update(1)
    call_no += 1
            
    t0 = time.time()

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

        if loss < 3:
            time.sleep(3)
            sparky.summarize_performance(model, firesdata[0, :, :, :])
            time.sleep(3)

        if loss == 0:
            print(""), print(""), print(""), print("GOT A PERFECT FIT : ")
            time.sleep(3)
            sparky.summarize_performance(model, firesdata[0, :, :, :])
            time.sleep(3)
            quit()
    
    xstash.append(parameters)
    ystash.append(loss)
    np.save(arr = xstash, file = "xstash.npy")
    np.save(arr = ystash, file = "ystash.npy")

    return loss

## create instance of model to make param counting easier
nparams = Model().nparams
print(f"nparams : {nparams}")
param_bounds = [(-2.0, 2.0) for i in range(nparams)]
param_bounds[nparams - 4] = (-100.0, 10.0)               ## allow much lower global intercept for spread rate
print(f"> initializing {optimizer.__name__} search over {nparams} variables with {n_calls} total calls")
call_no = 0
xstashh, ystashh = [list(stash) for stash in xstash], list(ystash)
if ((len(xstash) == 0) or (len(ystash) == 0)):
    xstashh, ystashh = None, None
opt = optimizer(objective_function, dimensions = param_bounds, n_calls = n_calls, n_initial_points = n_initial_points, x0 = xstashh, y0 = ystashh, acq_func = "gp_hedge", verbose = True)

print(f"OPTIMAL MODEL RESULTS : ")
for i in mini_batch_ids:
    print(f"optimal model's performance on mini-batch example {i}")
    sparky.summarize_performance(Model(parameters = opt.x), firesdata[i, :, :, :]) 
