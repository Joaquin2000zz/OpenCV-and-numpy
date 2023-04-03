#!/usr/bin/env python3
"""
module which contains compress_image function
"""
import io
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image


def compress_image(A, r, plot=False, gray_scale=False):
    """
    compress an image using SVD
    @A: image matrix to compress
    @r: range of significant values to take
    @plot: flag to determine whether plot the log and the
           cummulative sum of singular values or not
    @gray_scale: flag to determine whether the image would be
                 gray scaled or not 
    Returns: the compressed image
    """
    d = len(A.shape)
    w, h, c = A.shape
    if d == 3 and gray_scale:
        X = [np.mean(A, axis=-1)]
    elif d == 3:
        X = [A[:, :, i] for i in range(0, c)]
    X_approx = []
    for i, x in enumerate(X):
        # economy svd (returns first m columns of
        # U corresponding to the non zero singular values)
        U, S, VT = np.linalg.svd(x,
                                 full_matrices=False)
        S = np.diag(S)
        # always you have a new image, you must plot the information
        # from the image matrix to visualize the opptimal r
        if plot:
            plt.figure(1)
            plt.semilogy(S)
            plt.title(f'Singular Values log channel {i + 1}:')
            plt.figure(2)
            plt.plot(np.cumsum(S / np.sum(S)))
            plt.title(f'Singular Values: Cumulative Sum channel {i + 1}:')
            plt.show()
        X_approx.append(U[:, :r] @ S[:r, :r] @ VT[:r, :])
    return np.around(np.stack(X_approx, axis=-1)).astype(int)

if __name__ == '__main__':
    url = input('url from image to compress: ')
    response = requests.get(url=url)
    img = Image.open(io.BytesIO(response.content))
    A = np.asarray(img)
    plt.imshow(A)
    plt.title('Original image:')
    plt.axis('off')
    plt.show()
    R = input('r values: (note: you can use more than one r separating by .)\n')
    for i, r in enumerate(R.split('.')):
        plt.figure(i + 1)
        if i == 0:
            aproxA = compress_image(A, int(r), plot=True)
        else:
            aproxA = compress_image(A, int(r))
        plt.imshow(aproxA)
        plt.title(f'r: {r}')
    plt.axis('off')
    plt.show()
