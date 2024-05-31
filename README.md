# Introduction

## Problem Statement

Predicting the growth pattern and spread of wildfires for the next day
using a combination of time-series and regression models.

## Importance and Motivation

Accurate prediction of wildfire spread is crucial for effective disaster
management and resource allocation. The Spark Wildfire model, developed
by the Australian Research Data Commons, has been widely used for this
purpose. However, it uses a level set method to model spread
propagation, not machine learning. By leveraging the power of machine
learning techniques while incorporating the expertise of the Spark
Wildfire model, we aim to develop a model that can effectively forecast
the next-day growth and spread of wildfires, enabling better
preparedness and response efforts.  
We will be using the "Next Day Wildfire Spread" dataset from
[Kaggle](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread).

## Limitations of Existing Approaches

SPARK is a state-of-the-art wildfire prediction tool that uses a level
set method to propagate the fire. However, it does not use machine
learning techniques to do this. We see a big opportunity to use
convolution layers to build an accurate machine learning model for this
problem. By taking 2D features representing various properties of the
fire edge location (Wind, temperature, vegetation, etc.), we can train a
convolutional neural network to predict the next-day spread of a
wildfire, given these types of features.

## Contribution of this Project

Within the Spark model, a pivotal aspect of interest lies in its spread
rate calculation methodology. Although the accompanying paper () lacks
explicit details on how this calculation is performed for individual
grid cells, or if it undergoes temporal iteration updates within their
level-set method, it mentions that the spread rate is a function of the
environmental conditions at each grid point. This information alone
provides valuable insights, potentially serving as an additional set of
features for our model. However, there’s an opportunity to enhance this
component further. The attached paper does not indicate any optimization
of the spread rate function using training data. Incorporating such
optimization could promptly enhance the performance of the Spark model
while retaining its level-set method, which would be a significant
result in its own right. The mechanics of optimizing this function may
be too complex to take on, though, as the level-set method used is
governed by a differential equation in terms of spread rate, so there is
no obvious closed form expression of the updated fire boundary in terms
of the environmental conditions involved in the calculation of spread
rate. In this project, we will train a neural network to optimize the
spread rate function, to see if this kind of optimization can improve
prediction accuracy.

# Related Work

## Paper 1 - SPARK overview

[Link](https://www.researchgate.net/publication/272419229_SPARK_-_A_bushfire_spread_prediction_tool)  
This paper from 2015 details the SPARK Wildfire Tool and its ability to
predict wildfire spread. However, this model uses a level-set method,
where the user has to provide estimated spread rates for each grid
square in the fire map in order to simulate fire propagation. So, this
does not solve the problem that we want to solve, because SPARK does not
use machine learning in its calculations.  
As we do not have access to the SPARK tool API, we will simulate SPARK’s
level set method independently. As outlined in the paper above, we first
estimate the distance of each cell to the fire front *ϕ* (where
*ϕ*\[*i*,*j*\] \< 0 if cell (*i*,*j*) lies fully inside the fire), then
estimate the *spatial*[^1] gradient of the distance function, ∇*ϕ*, and
then update the fire front based on *δ**ϕ*

## Paper 2 - Wildfire Loss Functions

[Link](https://www.sciencedirect.com/science/article/pii/S1364815224000057?via%3Dihub)  
This paper examines and compares multiple different ways to measure loss
for spatial wildfire applications. It details various categories, such
as General Overlap, Partial Overlap, and Distance functions, which
measure the loss of the predicted 2D shape in different ways. This is
useful because it is related to our problem of predicting wildfire
spread, and it gives us insight into the best way to measure the loss of
our model.  
The loss function we used in our experiments was a weighted, smoothed
version of the area deviation index (ADI) referenced in the paper. This
weighting was based on the idea that the consequences of underestimating
fire spread greatly outweigh the consequences of overestimating it
(slightly). Specifically, using the notation from the paper above, we
used:
$$\\texttt{loss(\\texttt{pred}, true) = } \\frac{3\|\\texttt{pred}\_0 \\cap \\texttt{true}\_1\| + 2\|\\texttt{pred}\_1 \\cap \\texttt{true}\_0\|}{\|\\texttt{pred}\_1 \\cap \\texttt{true}\_1\|}$$
When the denominator there are two cases: either `pred` is just the zero
matrix - in which case we "smooth" the function by incrementing the
denominator by one, or our predicted fire completely missed the true
fire, in which case we just return a value which functions like
infinity[^2] in order to penalize the model heavily for this behavior.

## Paper 3 - Applying Bayesian Models to Wildfire Analysis

[Link](https://www.mdpi.com/2073-4433/14/3/559)  
In this paper, the researchers sought to reduce the computational
complexity of wildfire modelling systems through Bayesian models. They
found that without a drop in approximation accuracy, they were able to
reduce computational resource requirements by up to a factor of two.
While our proposal did not include any plans to use Bayesian inference,
this is very much a related problem and the findings are relevant,
especially if we run into a compute problem. We may use the findings of
this paper to improve the efficiency of our model, by building
incorporating Bayesian models into our optimization. Because we
constrained our model by forcing it to learn spread rates which are fed
into a level-set process instead of trying to learn the next fire front
directly, we were not able to exploit auto-differentiation or any
gradient-based optimization methods. As such, in order to optimize our
model we used `skopt.gp_minimize` which tries to model our loss function
as a multivariate Gaussian in order to generate good parameter guesses
to test. This is a form of Bayesian estimation.

# Description and Justification

## Dataset Details

The dataset we chose for this problem is [Next Day Wildfire
Spread](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread)
from Kaggle. This dataset includes 12 input features over a 64x64
kilometre region at 1 kilometre resolution. The features include: the
previous day fire mask (the location of the fire), elevation, wind
direction/speed, minimum/maximum temperatures, humidity and
precipitation, drought index, population density, vegetation, and an
energy release component. Due to the 2D nature of the data and our goal
of predicting the next-day wildfire spread, we believe we can apply
convolutional techniques to achieve strong performance on this
dataset.  
It is important to note that in our predictions, 1 means that a cell is
on fire, and 0 means that there is no fire in that cell.

## The SPARK Wildfire Model

The [SPARK Wildfire platform](https://research.csiro.au/spark/) is
developed by the CSIRO for Australian wildfire research and emergency
response teams. The paper linked in Section 2.1 details the Bushfire
Spread Prediction model that the SPARK platform uses.  
SPARK uses a level set method to model the propagation of the fire
perimeter over the landscape. The fire spread rate is governed by a
user-defined algebraic rate of spread model, which can incorporate
spatial variation in fuel properties and temporal variation in wind
speed/direction by randomly sampling from defined distributions during
simulations. By running an ensemble of simulations with different random
seeds, SPARK can generate probability of arrival maps, indicating the
likelihood of fire reaching different areas over a given time period.
The tool aims to provide an operational capability for real-time fire
spread modeling and risk assessment, while also enabling research into
complexities like environmental heterogeneity.

## Proposed Improvements

The SPARK model uses a computationally-intensive level set method to
simulate wildfires. We propose using machine learning models to optimize
the level set method and improve the accuracy in predicting next-day
wildfire spread.

### Cheaper Convolutions

In order to dramatically reduce the number of parameters we needed to
train while still harnessing the power of convolutions, we allowed the
model to learn one small convolution per feature, and then we performed
local regression with one shared weight per feature using both the
convolved and original values of the data.

# Experiments and Analysis

## Experiment 1 - SparkNet/SparseNet

### Results

These were our first attempts at building a model which worked in tandem
with a SPARK-like level set method to predict wildfire spread by
predicting spread rates for each cell. At this point we were still naive
and starry-eyed, so we set out to build the most flexible model
possible. Our first iteration was a 1-layer fully connected neural
network, dubbed ’sparknet’. Even with just one fully connected layer,
due to the sizes of our model input/output we needed
((64<sup>2</sup>)\*12) \* (64<sup>2</sup>) ∼ 200, 000, 000 parameters.
We quickly realized that optimizing a black box function with this many
parameters was intractable and scrapped this idea.  
Eager to hang onto the power of using a neural network, we decided to
try a mixed model consisting of a fully local regression component with
one weight per parameter and a neural network that encodes our
 ∼ 50, 000 into 3 activations (only requiring  ∼ 150, 000 parameters!).
We dubbed this one ’sparsenet’ and then set about trying to optimize it.
After a frustrating 12 hour attempt at optimizing this to no avail, we
scrapped this idea too.

## Experiment 2 - SparserNet

### Objective

Use learned convolutions in tandem with nonlinear local regression and
linear regression on global summary statistics (which means no hefty
global network) to build a flexible SPARK companion model with a
fraction of the number of parameters of earlier models.

### Methodology

In this iteration of our model, we are assuming that the spread rate in
a given cell is independent given feature values of itself and its 48
neighbor cells as well as the 36 global feature summary statistics.
After building a model with the components above (which required
approximately 700 parameters with default settings), we also set our
sights on a more reasonable testing goal - to prove that our model can
learn anything at all. In order to do this, we allowed our model to
optimize on one single fire at a time instead of calculating batch error
on stochastically selected mini-batches of fires. Ideally, our model
will overfit to this single example, illustrating that some kind of
learning is possible under the circumstances we’ve put our model under.

### Results

Even just optimizing our loss over a single fire with ten working
iterations of `gp_minimize` still took approximately an hour, so we were
only able to perform this experiment on two fires. The cropped
prediction of the optimal models on the fires they trained on is
visualized in Figure 1a (the rest of this specific visualization is
green 0’s). We have printed the predicted fire front and color-coded it
according to its correctness. Red cells indicate mislabeling an on-fire
cell as not on fire, yellow cells indicate the converse, and green cells
represent correct predictions. While the error is very high considering
our model is meant to be overfit to the provided fire, it is promising
that we were able to achieve the rough overlap of fire shapes that we
did with only 10 working iterations of `gp_minimize`. Modeling and
sampling a  ∼ 700-dimensional Gaussian is certain to be highly complex,
so truly optimizing this model would likely require working iterations
in the thousands, but these results are encouraging.

## Experiment 3 - SparkLin

### Objective

Try to build and train SPARK model which is entirely linear in order to
simplify search space and more readily permit model optimization (albeit
to the detriment of model flexibility).

### Methodology

In this iteration of the model, we are again assuming spread rates are
independent given global summary statistics and feature values in their
neighborhood. Additionally, though, we are making the strong assumption
that spread rates are linear with respect to environmental factors. To
achieve our objective we removed any non-linear activations from the
local and convolved regression which cut our total regression parameters
in half. Additionally we shrunk our learned convolutions from 7x7 to
5x5. In total, the default version of this model has a little over 350
parameters. Once again we sought to demonstrate the capability for
learning via overfitting to a single fire. The results of this
experiment are provided in Figure 1b in the same format as the results
for the SparserNet variation.

### Results

As can be immediately discerned from the results visualized in Figure
1b, SparkLin is not a flexible enough model to permit learning under
these circumstances. It’s likely that the problem of estimating spread
rates is not linear in terms of the environmental conditions, which
would contradict the assumption that SparkLin makes, and explain its
terrible performance.

![SparserNet’s Prediction Result](figs/sparsernet_cropped.jpg)

![SparkLin’s Prediction Result](figs/sparklin_full.jpg)

# Discussion and Future Work

## Main Conclusions

One of the largest takeaways we have from our work is that the breadth
of this problem is too large to optimize in a tractable number of
Bayesian optimization iterations.

## Strengths and Weaknesses

One strength of our approach is that, if we were able to achieve good
performance, we would not only yield a useful predictive model of
wildfire behavior but also a model of how different environmental
factors influence the spread rates of future fires. This knowledge would
not only help with predicting the movement of ongoing wildfires but also
aiding in decisions about where to deploy wildland fire resources
(controlled burns, fire lines, etc) to maximize their efficiency at
fighting fires before they even start. However, a glaring weakness of
our contribution is our limited amount of compute (Mark’s laptop does
not even have a GPU). Because we were training the models on our own
measly laptops and not at a designated data center, the training was
very slow and we had to carefully design the models to minimize the
number of parameters. If we had access to the necessary resources, we
think we could have experimented more efficiently and had time to find
the model architectures that perform the best on this problem.

## Future Directions

While we face a lack of compute issue, this problem has caught our
interest and we will continue to work on it beyond the deadline of this
project. Our current ideas to improve our model include the following
changes: trying to restructure our problem such that we can find a
auto-differentiable function that performs an iteration of our
SPARK-like level set method so that we can apply gradient-based
optimization, adding flexibility so that the SPARK-like level set
iteration can learn how to handle/interpret corrupted/cloud-obscured
data (-1 entries) correctly which it currently cannot, adding more
specific fixed convolutions (e.g. a hill/valley detector) for certain
layers to further encourage the model on what to learn and reducing the
number of nooks and crannies in our optimization space for it to get
stuck in, and finally adding a time element to the gradient of our
distance function so that we can train our model on multiple sequential
snapshots of the same fire. Additionally, as a point of curiosity, we
want to see how a boiler plate fully connected neural network would
perform if we allowed it to learn the next fire evolution directly
instead of forcing it to act only via the spread rates in a SPARK-like
level set method.

<div class="thebibliography">

50 Miller, Claire & Hilton, J.E. & Sullivan, Andrew & Prakash, Mahesh.
(2015). SPARK – A bushfire spread prediction tool. IFIP Advances in
Information and Communication Technology. 448. 262-271.
10.1007/978-3-319-15994-2_26.

</div>

[^1]: The original paper was not very specific with its notation, so

[^2]: As our objective function must return float values in order to
    conform to the requirements of our optimizer, `skopt.gp_minimize`,
    we use $"\\infty" = 4 \~\\times\~ \\texttt{np.prod(pred.shape)}$,
    which ’acts like infinity’ in the sense that
    $$"\\infty" \> \\texttt{loss(pred, true)} \\forall \\texttt{(loss, pred)}\~:\~\|\\texttt{true}\_1 \\cap \\texttt{pred}\_1\| = 0)\~\\land\~ (\|\\texttt{true}\_1\| \\neq 0)$$