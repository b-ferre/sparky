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
    distances = np.zeros_like(matrix, dtype=int)
    queue = deque()
    distances[borders == 1] = 0
    queue.extend(np.transpose(np.nonzero(borders == 1)))
    
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < rows) and (0 <= ny < cols) and (distances[nx, ny] == 0):
                if borders[nx, ny] == 0:
                    distances[nx, ny] = distances[x, y] + 1
                    queue.append((nx, ny))
    
    # Replace all internal cell distances with their negation
    distances[np.logical_and(matrix == 1, np.all(borders[1:-1, 1:-1] == 0, axis=(0, 1)))] *= -1
    
    # Set cells outside the fire-front to 0
    distances[matrix == 0] = 0
    
    return distances

## helper that, given a numerical matrix with entries representing function values,
## estimates the gradient of said function at every point
def estimate_gradient(matrix):
    matrix = np.matrix(matrix)
    rows, cols = matrix.shape
    gradient_x = np.zeros((rows, cols))
    gradient_y = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            if i > 0:
                gradient_x[i, j] += (float(matrix[i, j] or 0) - float(matrix[i - 1, j] or 0))
            if i < rows - 1:
                gradient_x[i, j] += (float(matrix[i + 1, j] or 0) - float(matrix[i, j] or 0))
            if j > 0:
                gradient_y[i, j] += (float(matrix[i, j] or 0) - float(matrix[i, j - 1] or 0))
            if j < cols - 1:
                gradient_y[i, j] += (float(matrix[i, j + 1] or 0) - float(matrix[i, j] or 0))

    return gradient_x, gradient_y

## prints front matrix with one row per line and all entries evenly spaced
def print_matrix(matrix):
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

###############################################################################################################
##                                              MODEL CODE                                                   ##
###############################################################################################################

ones = np.ones((10,10))
print_matrix(dist_to_front(ones))