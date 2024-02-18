# Linear Regression Model with Multiple features/variables
# Here, Solution is given using loops

import copy
import numpy as np
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

# solution using vectorisation 
# In numpy, '@' and matmul represent matrix multiplication

# calculating model's cost 
def model_cost(x, y, w, b):
    m, n = x.shape
    f_wb = x @ w + b
    cost = np.sum((f_wb - y)**2) / (2 * m)
    print(cost)
    return cost

# calculation gradient using vectorisation 
def gradient(x, y, w, b):
    m = x.shape[0]
    f_wb = x @ w + b
    err = f_wb - y # error on prediction
    dj_dw = (x.T @ err) / m
    dj_db = np.sum(err) / m
    return dj_dw, dj_db

def gradient_descent(x_in, y_out, w, b, num_iters, alpha):
    w_temp = copy.deepcopy(w)
    b_temp = b
    for i in range(num_iters):
        dj_dw, dj_db = gradient(x_in, y_out, w_temp, b_temp)
        w_temp = w_temp - dj_dw * alpha
        b_temp = b_temp - dj_db * alpha
    return w_temp, b_temp

x_in = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_out = np.array([460, 232, 178])
b_initial = 0.
w_initial = np.zeros((4,))
num_iters = 1000
alpha = 5.0e-7
w_final, b_final = gradient_descent(x_in, y_out, w_initial, b_initial, num_iters, alpha)
print(f"The value of w and b after gradient is ({w_final}, {b_final:0.2f})")


