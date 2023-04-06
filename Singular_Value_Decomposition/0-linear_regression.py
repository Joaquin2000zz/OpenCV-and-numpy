#!/usr/bin/env python3
"""
module which contains the first part of least squares regression section
and contains a nice introduction to what least squares regression is.
linear regression is basically a way of find the estimate value of a variable
given one more variables. matematically, we start with:
Ax = b our goal is find the solution for x. so the first step is decompose A in their singular values
such that A = U S VT. now we can rewrite the equation in terms of the SVD of A such that:
U S VT x = b. we multiply by UT and V in both sides of the equation and by S^-1 such that:
V S^-1 UT U S VT x = V S^-1 UT b -> (UT U = I, S^-1 S = I, V VT = I) -> x = V S^-1 UT b
because we're using the economy SVD, the values are approximations and x is xtilde.
"""
import matplotlib.pyplot as plt
import numpy as np

# before use a model, a common practice is to test it in some task
# that we know the answer in order to se how good the model is
x = 3 # True slope
a = np.arange(-2, 2, .25).reshape(-1, 1)
b = a * x + np.random.randn(*a.shape) # adding noise to simulate normal data variance

plt.plot(a, a * x, color='k', linewidth=2, label='True Line')
plt.plot(a, b, 'x', color='r', markersize=10, label='Noisy Data Points')

U, S, VT = np.linalg.svd(a, full_matrices=False)
xtilde = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ b

plt.plot(a, a * xtilde, '--', color='b', linewidth=4, label='Regression Line')
plt.xlabel('a')
plt.xlabel('b')
plt.grid(linestyle='--')
plt.legend()
plt.show()
