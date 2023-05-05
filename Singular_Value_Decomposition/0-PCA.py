#!/usr/bin/env python3
"""
first, to see how it works, we gonna test the PCA in a toy dataset
i.e. a dataset in which we know the answer to see how it actually works
I studied this subject in Holberton machine learning specialization before,
but I think that following this course is a great way to recall and also to
get a deep intuition of what PCA is. 
"""
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = [16, 8]
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('PCA Gaussian')
xC = np.array([2, 1]) # center of data (mean)
sig = np.array([2, .5]) # principal axes

theta = np.pi / 3 # rotate cloud

R = np.array([[np.cos(theta), -np.sin(theta)], # rotation matrix
              [np.sin(theta), np.cos(theta)]])

nPoints = 10000 # create 10,000 points
# stretching out in an ellipsoid in a factor of two, smashing in a factor of .5,
# and rotating by a factor of pi / 3 our initial gaussian cloud to obtain a cloud of data
# that is not at the origin. the objective is to compute the SVD to show how the SVD recovers
# its principal axis of maximum and minimum variance and also, how allows us to delimite some sigma confidence intervals
# to determine how likely a new datapoint is to belong to this distribution 
X = R @ np.diag(sig) @ np.random.randn(2,
                                       nPoints) + np.diag(xC) @ np.ones(shape=(xC.shape[0],
                                                                                  nPoints))

ax1.plot(X[0, :], X[1, :], '.', color='k')
ax1.grid()
plt.xlim((-6, 8))
plt.ylim((-6, 8))

# mean vector. aggregates all dimensions across all data points
Xbar = np.mean(X, axis=1, keepdims=True)

B = X - Xbar # mean subtracted data (bringing back to the origin)

# find principal componets SVD
# U will tell us the rotation and S the variance in each direction
U, S, VT = np.linalg.svd(B / np.sqrt(nPoints), full_matrices=False)

ax2.plot(X[0, :], X[1, :], '.', color='k') # plot data to overlay PCA
ax2.grid()
plt.xlim((-6, 8))
plt.ylim((-6, 8))

theta = 2 * np.pi * np.arange(0, 1, .01)

# 1-std confidence interval
Xstd = U @ np.diag(S) @ np.array([np.cos(theta),
                                  np.sin(theta)])

# plot std intervals
ax2.plot(Xbar[0] + Xstd[0, :], Xbar[1] + Xstd[1, :],
         color='r', linewidth=1)
ax2.plot(Xbar[0] + 2 * Xstd[0, :], Xbar[1] + 2 * Xstd[1, :],
         color='r', linewidth=1)
ax2.plot(Xbar[0] + 3 * Xstd[0, :], Xbar[1] + 3 * Xstd[1, :],
         color='r', linewidth=1)
# plot principal components U[:, 0]S[0] and U[:, 1]S[1]
ax2.plot(np.array([Xbar[0], Xbar[0] + U[0, 0] * S[0]]),
         np.array([Xbar[1], Xbar[1] + U[1, 0] * S[0]]), '-', color='cyan',
         linewidth=1)
ax2.plot(np.array([Xbar[0], Xbar[0] + U[0, 1] * S[1]]),
         np.array([Xbar[1], Xbar[1] + U[1, 1] * S[1]]), '-', color='cyan',
         linewidth=1)

plt.show()
