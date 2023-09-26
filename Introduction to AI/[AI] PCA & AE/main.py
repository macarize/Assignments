import numpy as np
from sklearn.preprocessing import normalize
def mean(x, axis=None):
    if axis is not None:
        axis_len = x.shape[0]
        mean = np.array([])
        axis_sum = np.sum(x, axis=axis)
        axis_mean = axis_sum / axis_len
        mean = np.append(mean, axis_mean)
        return mean
    else:
        return x.sum()/x.size
def std(x, axis=None):
    x_ = x
    if axis is None:
        x_mean = mean(x_)
        return np.sqrt(np.sum((x_-x_mean)**2)/(x.size-1))
    else:
        x_mean = mean(x, axis_=axis)
        print(x.shape[0])
        return np.sqrt(np.sum((x_- x_mean)**2, axis=axis)/(x.shape[0]-1))

def normalize_(x, axis=None):
    if axis is not None:
        l2_norm = np.sqrt(np.sum(x * x, axis=axis))
        return x / l2_norm
    else:
        l2_norm = np.sqrt(np.sum(x * x))
        return x / l2_norm(x)

def SVD(matrix):
    ATA = np.matmul(matrix.T, matrix)

    n = ATA.shape[0]
    eye = np.eye(n)
    sum_eigenvalues = np.sum(np.matmul(ATA, eye))
    print(sum_eigenvalues)
    mul_eigenvalues = np.linalg.det(ATA)
    print(mul_eigenvalues)

f_value = np.loadtxt("f_value.txt", delimiter=',')
min = np.amin(f_value)
f_value += np.abs(min)
f_value = normalize_(f_value, axis=0)
print(SVD(f_value))