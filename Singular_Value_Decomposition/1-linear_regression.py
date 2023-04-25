#!/usr/bin/env python3
"""
now we gonna go through an example of a linear regression in which we will have more than
one singular factor a (as in the previous regression) to build the prediction of b
links: - https://github.com/dynamicslab/databook_python/blob/master/DATA/hald_heat.csv
       - https://github.com/dynamicslab/databook_python/blob/master/DATA/hald_ingredients.csv
"""
import matplotlib.pyplot as plt
import numpy as np


A = np.genfromtxt('DATA/hald_ingredients.csv', delimiter=',')
b = np.genfromtxt('DATA/hald_heat.csv', delimiter=',')

# normally the df is splitted into training and testing sets, but just for visualization,
# we will use the entire dataset (not recommended)
U, S, VT = np.linalg.svd(A, full_matrices=False)
xtilde = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ b

plt.plot(b, color='k', linewidth=2, label='Summary')
plt.plot(A @ xtilde, '-o', color='r', linewidth=1.5, markersize=6, label='Regression')
plt.grid(linestyle='--')
plt.legend()
plt.show()