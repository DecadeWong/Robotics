import numpy as np
import math
#three by three  covariance matrix
def cov_matrix(data):
    R_mean = sum(data[0])/len(data[0])
    G_mean = sum(data[1])/len(data[1])
    B_mean = sum(data[2])/len(data[2])

    mean = [[R_mean],[G_mean],[B_mean]]
    RGB = np.matrix(data)
    dimension = RGB.shape #find the dimension of that matrix
    # scanning column and each column with three rows
    val = [(RGB[:,i] - mean) * (RGB[:,i] -mean).T for i in range(dimension[1])] #find number of column
    cov_matrix = sum(val)/(dimension[1]-1)
    return cov_matrix, mean


def G_probability (mean, cov, RGB):
    '''gaussian probability for one 3*1 vector'''
    val = -0.5*(RGB-mean).T * np.linalg.inv(cov) *(RGB-mean)
    A_D = abs(np.linalg.det (cov)) ** 0.5
    P = 1/(A_D*(2*math.pi) ** 1.5) * math.exp(val)
    return P

def linear_regression(area, distance): # row vector as input
    v_distance = np.matrix(distance).T

    v_area = np.matrix(area).T
    ones = np.ones((len(area), 1))
    A = np.concatenate((v_area, ones), axis=1)
    parameter = (A.T * A).I *A.T * v_distance

    return parameter #ax+b it is vector a, b



