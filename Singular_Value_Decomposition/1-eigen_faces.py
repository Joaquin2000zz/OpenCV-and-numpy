import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as sci

mat_contents = sci.loadmat(os.path.join('..', 'DATA', 'allFaces.mat'))
faces = mat_contents['faces']
n = int(mat_contents['n'])
m = int(mat_contents['m'])
nFaces = np.ndarray.flatten(mat_contents['nfaces'])
# each image is composed by a column vector
# taking the first 36 people for training data
# and computing the mean for each pixel in the images
# thus, cumputing the average face
trainingFaces = faces[:, :nFaces[:36].sum()]
avgFace = trainingFaces.mean(axis=1)
X = trainingFaces - avgFace[:, np.newaxis]
# computing the singular value decomposition from
# the mean centred images (X matrix) i.e. eigen values and vectors
# and also the PCA
U, S, VT = np.linalg.svd(X, full_matrices=None)
# plotting the avg face and the eigen faces i.e. eigen vectors from the X matrix
fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
img_avg = ax1.imshow(avgFace.reshape((m, n)).T)
img_avg.set_cmap('gray')
plt.axis('off')
plt.title('Avg Face')
ax2 = fig1.add_subplot(122)
eigen_face = ax2.imshow(U[:, 0].reshape((m, n)).T)
eigen_face.set_cmap('gray')
plt.axis('off')
plt.title('Eigen Face')
plt.show()

# Now we gonna reconstruct a sample image that was not in the training set
# by projecting the mean centred image into the eigen face space, computing its
# inner product with r columns of U

sample_person = faces[:, nFaces[:36].sum()]
plt.imshow(sample_person.reshape((m, n)).T)
plt.title('Original Image')
plt.axis('off')

sample_person = sample_person - avgFace

r_list = [25, 50, 100, 200, 400, 800, 1600]

for r in r_list:
    recon_face = avgFace + U[:, :r] @ U[:, :r].T @ sample_person
    plt.imshow(recon_face.reshape((m, n)).T)
    plt.title(f'Reconstructed face with a factor of r = {r}')
    plt.axis('off')

plt.show()
