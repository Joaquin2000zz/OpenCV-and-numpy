import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

obs = np.loadtxt(os.path.join('..', 'DATA',
                              'ovariancancer_obs.csv'), delimiter=',')

with open(os.path.join('..', 'DATA',
                       'ovariancancer_grp.csv'), 'r', encoding='utf-8') as f:
    grp = f.read().split('\n')

U, S, VT = np.linalg.svd(obs, full_matrices=None)

fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
ax1.semilogy(S, '-o', color='k')
ax2 = fig1.add_subplot(122)
ax2.plot(np.cumsum(S) / np.sum(S), '-o', color='k')

fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')

has_legend_c = False
has_legend_n = False
for i in range(obs.shape[0]):
    x = VT[0, :].dot(obs[i, :])
    y = VT[1, :].dot(obs[i, :])
    z = VT[2, :].dot(obs[i, :])

    if grp[i] == 'Cancer':
        if not has_legend_c:
            ax.scatter(x, y, z, marker='x', color='r', s=50, label='Cancer')
            has_legend_c = True
        ax.scatter(x, y, z, marker='x', color='r', s=50)
    else:
        if not has_legend_n:
            ax.scatter(x, y, z, marker='o', color='b', s=50, label='No Cancer')
            has_legend_n = True
        ax.scatter(x, y, z, marker='o', color='b', s=50)

ax.view_init(25, 20)
plt.legend()
plt.show()
