import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as sci

mat_contents = sci.loadmat(os.path.join('..', 'DATA', 'allFaces.mat'))
faces = mat_contents['faces']

nFaces = np.ndarray.flatten(mat_contents['nfaces'])
trainingFaces = faces[:, :nFaces[:36].sum()]
avgFace = trainingFaces.mean(axis=1)
X = trainingFaces - avgFace[:, np.newaxis]
U, S, VT = np.linalg.svd(X ,full_matrices=False)

P1num = 2
P2num = 7

P1 = faces[:, nFaces[:P1num - 1].sum(): nFaces[:P1num].sum()]
P2 = faces[:, nFaces[:P2num - 1].sum(): nFaces[:P2num].sum()]
P1 = P1 - avgFace[:, np.newaxis]
P2 = P2 - avgFace[:, np.newaxis]

# theese modes are chosen because when we talk in the context of clustering,
# we want the PC columns which had less energy of information which is the same in all people
# the firsts principal component columns will have most important information (everyone has a nose
# has a oval head, mounth, etc). in contrast, the nexts modes will have information related
# with the geometry of each persons's face features giving us the possibility of discern between P1 and P2
PCAmodes = [5, 6] # Project onto PCA modes 5 and 6
Ur = U[:, PCAmodes - np.ones_like(PCAmodes)].T
PCACoords1 = Ur @ P1
PCACoords2 = Ur @ P2
print('PCACoords1.shape', PCACoords1.shape)
plt.plot(PCACoords1[0, :], PCACoords1[1, :], 'd', color='k',label='person1')
plt.plot(PCACoords2[0, :], PCACoords2[1, :], '*', color='r',label='person2')
plt.legend()
plt.show()
