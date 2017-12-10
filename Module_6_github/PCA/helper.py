import numpy as np
from matplotlib import pyplot as plt

def normalize(X):
    '''
          Normalise data before processing
    '''
    num = X.shape[1]

    NormParams = np.zeros((2, num))
    NormParams[0] = X.mean(axis=0)
    NormParams[1] = X.std(axis=0)

    X = (X - NormParams[0]) / NormParams[1]

    return X, NormParams

def transform(X, num):
    '''
            Select components have largest variance
    '''
    cov = np.dot(X.T, X) / len(X)
    e_val, e_vect = np.linalg.eig(cov)

    e_val = np.absolute(e_val)

    ind = np.argsort(-e_val)
    e_vect = e_vect[:, ind]
    e_vect = e_vect.astype(float)
    evect_reduced = e_vect[:, :num]
    new_X = np.dot(X, evect_reduced)
    return new_X, evect_reduced


def restore(X_reduced, evect_reduced, norm_params):
    '''
        Restore "original" values:
            1) Original size
            2) Rescale
    '''
    restored = np.dot(X_reduced, evect_reduced.T)
    for j in range((restored).shape[1]):
        restored[:, j] = (restored[:, j] * norm_params[1, j]) + norm_params[0][j]
    return restored

points = 10
X = np.zeros((points,2))
x = np.arange(1,points+1)
y = 4 * x *x + np.random.randn(points)*2
X[:,1] = y
X[:,0] = x
number_of_components = 1

# normalization
X_norm, norm_params = normalize(np.copy(X))

# dimension reduction
X_reduced, evect_reduced = transform(X_norm, number_of_components)

# restoring dimensions
restored_X = restore(X_reduced, evect_reduced,norm_params )

plt.figure()
plt.scatter(X[:, 0], X[:, 1], color='c', label='Initial')
plt.scatter(restored_X[:, 0], restored_X[:, 1], color='y', label='Restored')
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# from PIL import Image
#
# number_of_components_image = 50
#
# img = Image.open('pct.jpg')
#
# img = img.convert('L')
#
# img_X = (np.copy(np.asarray(img))).astype(float)
#
# X_norm_img, norm_params = normalize(img_X)
#
# X_reduced_img, evect_reduced = transform(X_norm_img, number_of_components_image)
#
# X_restored_img = restore(X_reduced_img, evect_reduced, norm_params)
#
# restored_img = Image.fromarray(X_restored_img.astype(int))
#
# img.show()
# restored_img.show()