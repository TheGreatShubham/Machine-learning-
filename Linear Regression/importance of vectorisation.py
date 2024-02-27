import numpy as np
import time

def my_dot(a, b):
    ans = 0
    m = a.shape[0]
    for i in range(m):
        ans = ans + (a[i] * b[i])
    return ans

a = np.random.rand(10000000)
b = np.random.rand(10000000)
start = time.time()
result = my_dot(a, b)
end = time.time()
print(f"The value of dot product without vectorisation: {result}")
print(f"The value of dot product using loop: {1000*(end - start):.4f} ms")

start = time.time()
result = np.dot(a, b)
end = time.time()
print(f"The value of dot product using vectorisation: {result}")
print(f"The value of dot product using vectorisation: {1000*(end - start):.4f} ms")
