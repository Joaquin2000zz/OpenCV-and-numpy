#!/usr/bin/env python3
"""
now we'll apply a linear regression applying the splitting of the data
into train and test set from random sampling by shuffling the rows for
help to improve the generalization of the data of the dataset
link: https://github.com/dynamicslab/databook_python/blob/master/DATA/housing.data
"""
import matplotlib.pyplot as plt
import numpy as np


data = np.loadtxt('DATA/housing_data.csv')
n = data.shape[0]
A = data[:, :-1]
b = data[:, -1]
A = np.pad(A, [(0, 0), (0, 1)], mode='constant', constant_values=1)

# significance factor of each variable to the home value
A_mean = np.mean(A, axis=0).reshape(-1, 1)
A2 = A - np.ones(shape=(A.shape[0], 1)) @ A_mean.T
for i in range(A.shape[1] - 1):
    A2std = np.std(A2[:, i])
    A2[:, i] = A2[:, i] / A2std
A2[:, -1] = np.ones(A.shape[0])

U, S, VT = np.linalg.svd(A2, full_matrices=False)
xtilde = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ b
x_tick = range(xtilde.shape[0] - 1) + np.ones(xtilde.shape[0] - 1)
plt.bar(x_tick, xtilde[: -1])
plt.xlabel('Attribute')
plt.ylabel('Significance')
plt.xticks(x_tick)
plt.show()

shuffle = np.random.permutation(n)
data = data[shuffle, :]
A, b = data[:, :-1], data[:, -1]  # A = other values, b = housing values
# pad for non zero offset
split = int(n * .8)


A_train, A_test = A[split:, :], A[:split, :]
b_train, b_test = b[split:], b[:split]

U, S, VT = np.linalg.svd(A_train, full_matrices=False)
xtilde = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ b_train

plt.plot(b_train, color='k', linewidth=2, label='True Line')
plt.plot(A_train @ xtilde, '-o', color='r', linewidth=1,
         markersize=6, label='Regression Training Set')
plt.grid(linestyle='--')
plt.legend()
plt.show()
plt.plot(b_test, color='k', linewidth=2, label='True Line')
plt.plot(A_test @ xtilde, '-o', color='r', linewidth=1,
         markersize=6, label='Regression Test Set')
plt.grid(linestyle='--')
plt.legend()
plt.show()
