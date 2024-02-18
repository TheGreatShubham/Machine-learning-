import copy
import numpy as np
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

def my_dot(w, x):
    m = x.shape[0]
    f_wb = 0
    for i in range(m):
        f_wb = f_wb + (x[i] * w[i])
    return f_wb

def model_cost(x_in, y_out, w, b):
    m = x_in.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb = my_dot(x_in[i], w) + b
        cost = cost + (f_wb - y_out[i])**2
    total_cost = cost / (2 * m)
    return total_cost

def gradient(x_in, y_out, w, b):
    m, n = x_in.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range(m):
        f_wb = my_dot(x_in[i], w) + b
        for j in range(n):
            dj_dw[j] = dj_dw[j] + (f_wb - y_out[i]) * x_in[i,j]
        dj_db = dj_db + (f_wb - y_out[i])
    dj_db = dj_db / m
    dj_dw = dj_dw / m
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