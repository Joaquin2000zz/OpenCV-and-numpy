#!/usr/bin/env python3
"""
demonstrating how SVD preserves the information
in rotations and scaling from angles of a original matrix
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d


theta = np.array([np.pi / 15, -np.pi / 9, -np.pi / 20]) # angles
Sigma = np.diag([3, 1, .5]) # streching out first dimension and smashing third dimension

# rotation matrix with respect x axis
Rx = np.array([[1, 0, 0],
               [0, np.cos(theta[0]), -np.sin(theta[0])],
               [0, np.sin(theta[0]), np.cos(theta[0])]])
# rotation matrix with respect y axis
Ry = np.array([[np.cos(theta[1]), 0, -np.sin(theta[1])],
               [0, 1, 0],
               [np.sin(theta[1]), 0, np.cos(theta[1])]])
# rotation matrix with respect z axis
Rz = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
               [np.sin(theta[2]), np.cos(theta[2]), 0],
               [0, 0, 1]])
# rotating and scaling
X = Rx @ Ry @ Rz @ Sigma

# we decided to build the rotation and scaling
# but in the case you first have X, you can appreciate how by computing the SVD(X)
# and multiplying the U times the Sigma diagonal, you would obtain the same result.
# also notice that the VT is also the identity matrix
# (delete comment in both lines below to confirm)
# U, S, VT = np.linalg.svd(X)
# X = U @ np.diag(S)
# plotting sphere
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
u = np.linspace(-np.pi, np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

# plotting the surface
surf1 = ax1.plot_surface(x, y, z,
                         cmap='jet', alpha=.6, facecolors=plt.cm.jet(z),
                         linewidth=.5, rcount=30, ccount=30)
surf1.set_edgecolor('k')
ax1.set_xlim3d(-2, 3)
ax1.set_ylim3d(-2, 3)
ax1.set_zlim3d(-2, 3)

xR = np.zeros_like(x)
yR = np.zeros_like(y)
zR = np.zeros_like(z)

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        vec = [x[i, j], y[i, j], z[i, j]]
        vecR = X @ vec
        xR[i, j] = vecR[0]
        yR[i, j] = vecR[1]
        zR[i, j] = vecR[2]
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(xR, yR, zR,
                         cmap='jet', alpha=.6, facecolors=plt.cm.jet(z),
                         linewidth=.5, rcount=30, ccount=30)
surf2.set_edgecolor('k')
ax2.set_xlim3d(-2, 3)
ax2.set_ylim3d(-2, 3)
ax2.set_zlim3d(-2, 3)
plt.show()