
'''

Implementation of the Mahalanobis Distance as a metric between two data cubes
Function has been
http://nbviewer.ipython.org/gist/kevindavenport/7771325

'''

import numpy as np

def MahalanobisDist(x, y):
    assert x.shape == y.shape
    shape = x.shape
    x = x.ravel()
    y = y.ravel()
    covariance_xy = np.cov(x,y, rowvar=0)
    inv_covariance_xy = np.linalg.inv(covariance_xy)
    xy_mean = np.mean(x),np.mean(y)
    x_diff = np.array([x_i - xy_mean[0] for x_i in x])
    y_diff = np.array([y_i - xy_mean[1] for y_i in y])
    diff_xy = np.transpose([x_diff, y_diff])

    md = np.empty((diff_xy.shape[0],1))
    for i in range(len(diff_xy)):
        md[i,:] = np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]),inv_covariance_xy),diff_xy[i]))
    return md.reshape(shape)