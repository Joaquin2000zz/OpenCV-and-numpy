import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as sci

mat_contents = sci.loadmat(os.path.join('..', 'DATA', 'allFaces.mat'))
faces = mat_contents['faces']
m = int(mat_contents['m'])
n = int(mat_contents['n'])
nfaces = np.ndarray.flatten(mat_contents['nfaces'])

allPersons = np.zeros((n * 6, m * 6))
count = 0

for i in range(6):
    for j in range(6):
        allPersons[i * n: (i + 1) * n, j * m: (j + 1) * m] = faces[:, nfaces[:count].sum()].reshape((m, n)).T
        count += 1
img = plt.imshow(allPersons)
img.set_cmap('gray')
plt.axis('off')
plt.show()

for person in range(nfaces.shape[0]):
    subset = faces[:, nfaces[: person].sum(): nfaces[: person + 1].sum()]
    allFaces = np.zeros((n * 8, m * 8))

    count = 0

    for i in range(8):
        for j in range(8):
            if count < nfaces[person]:
                allFaces[i * n: (i + 1) * n, j * m: (j + 1) * m] = subset[:, count].reshape((m, n)).T
                count += 1
    img = plt.imshow(allFaces)
    img.set_cmap('gray')
    plt.axis('off')
    plt.show()
