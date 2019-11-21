#cython: language_level=3
'''
Created on Tue Oct 21 11:29:07 2019
@coded by: yudhiprabowo
'''

cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport sqrt
from libc.stdio cimport printf

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef unsigned char[:,::1] kmeans_calc(unsigned char[:,:,::1] img, int row, int col, int k):
    cdef:
        int i, j, d, idk, a
        double dmin, distm, dist, temp
        unsigned char[::1] dn
        unsigned char[:,::1] kmeans
        double[:,::1] mean, accum
    
    kmeans = np.zeros((row, col), dtype=np.uint8)
    mean = np.zeros((k, 3), dtype=np.double)
    accum = np.zeros((k, 4), dtype=np.double)
    dn = np.zeros(k, dtype=np.uint8)
    
    distm = 0
    for i in range(k):
        mean[i, 0] = int(float(i+1)/(k+1)*256)
        mean[i, 1] = int(float(i+1)/(k+1)*256)
        mean[i, 2] = int(float(i+1)/(k+1)*256)
        distm += mean[i, 0]**2 + mean[i, 1]**2 + mean[i, 2]**2
        dn[i] = int(i/float(k-1)*255)
    distm = distm//(3*k)
    temp = 0
    while(distm != temp):
        temp = distm
        accum[:, :] = 0
        for i in range(row):
            for j in range(col):
                for d in range(k):
                    dist = sqrt((mean[d, 0] - img[0, i, j])**2 + (mean[d, 1] - img[1, i, j])**2 + (mean[d, 2] - img[2, i, j])**2)
                    if(d == 0):
                        dmin = dist
                        idk = d
                    if(d > 0 and dist < dmin):
                        dmin = dist
                        idk = d
                kmeans[i, j] = dn[idk]
                accum[idk, 0] += img[0, i, j]
                accum[idk, 1] += img[1, i, j]
                accum[idk, 2] += img[2, i, j]
                accum[idk, 3] += 1
        distm = 0
        for i in range(k):
            mean[i, 0] = int(accum[i, 0]//accum[i, 3])
            mean[i, 1] = int(accum[i, 1]//accum[i, 3])
            mean[i, 2] = int(accum[i, 2]//accum[i, 3])
            distm += mean[i, 0]**2 + mean[i, 1]**2 + mean[i, 2]**2
        distm = distm//(3*k)
        printf("%lf %lf\n", distm, temp)
    
    return kmeans
