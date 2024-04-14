import numpy as np
from collections import deque

###############################################################################################################
##                                                HELPERS                                                    ##
###############################################################################################################

## helper which uses BFS to generate distance matrix from fire-front matrix
def dist_to_front(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    distances = [[float('inf')] * cols for _ in range(rows)]
    queue = deque()

    # Find all cells with value 1 and add them to the queue
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 1:
                distances[i][j] = 0
                queue.append((i, j))

    while queue:
        x, y = queue.popleft()
        # Check all 8 adjacent cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                # Check if the adjacent cell is within bounds and update the distance
                if 0 <= nx < rows and 0 <= ny < cols and distances[nx][ny] == float('inf'):
                    distances[nx][ny] = distances[x][y] + 1
                    queue.append((nx, ny))

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
                gradient_x[i, j] += (matrix[i, j] - matrix[i - 1, j])
            if i < rows - 1:
                gradient_x[i, j] += (matrix[i + 1, j] - matrix[i, j])
            if j > 0:
                gradient_y[i, j] += (matrix[i, j] - matrix[i, j - 1])
            if j < cols - 1:
                gradient_y[i, j] += (matrix[i, j + 1] - matrix[i, j])

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


###############################################################################################################
##                                              MODEL CODE                                                   ##
###############################################################################################################


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

    ## update front to reflect new distances
    new_front[new_phi < 1] = 1

    return(new_front)

## example data (25 x 25) where every cell has spread rate 1 and fire starts as a single central point
ex_front = np.zeros((25, 25))
ex_front[12, 12] = 1
ex_spread = np.ones((25, 25)) / 2


print_matrix(ex_spread)
spark_iter(ex_front, ex_spread)