# Linear Regression for a single feature / variable 
import copy
import numpy as np 

# Cost function (Mean Squared Error)
def model_cost(w, b, x_in, y): 
    m = x_in.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w * x_in[i] + b
        cost_sum = cost_sum + (f_wb - y[i])**2
    total_cost = (1 / (2 * m)) * cost_sum 
    return total_cost

# Gradient of the cost function with respect to weights (w) and bias (b)
def gradient(w, b, y, x_in):
    m = x_in.shape[0]
    df_dw = 0
    df_db = 0
    for i in range(m):
        f_wb = w * x_in[i] + b
        df_dw = df_dw + ((f_wb - y[i]) * x_in[i])
        df_db = df_db + (f_wb - y[i])
    df_dw = df_dw / m
    df_db = df_db / m
    return df_dw, df_db

# Gradient Descent algorithm to update weights and bias
def gradient_descent(x_in, y, num_iters, w, b, alpha):
    w_temp = copy.deepcopy(w) # avoid modifying global w_in
    w_temp = w;
    b_temp = b
    for i in range(num_iters):
        df_dw, df_db = gradient(w_temp, b_temp, y, x_in)
        w_temp = w_temp - alpha * df_dw
        b_temp = b_temp - alpha * df_db
    return w_temp, b_temp

# Input features and output values
x_in = np.array([1.0, 2.0])   #features 
y_out = np.array([300.0, 500.0])   #output value

# Initial weights and bias
w_initial = 0
b_initial = 0

# Calculate and print the initial cost
ans = model_cost(w_initial, b_initial, x_in, y_out)
print(f"The cost before gradient descent: {ans}")

# Set hyperparameters
num_iters = 10000
alpha = 1.0e-2

# Apply gradient descent to update weights and bias
w_final, b_final = gradient_descent(x_in, y_out, num_iters, w_initial, b_initial, alpha)
print(f"The vlaue of w and b after gradient descent: ({w_final:8.4f},{b_final:8.4f})")

# Calculate and print the final cost
ans = model_cost(w_final, b_final, x_in, y_out)
print(f"The cost after gradient descent: {ans}")