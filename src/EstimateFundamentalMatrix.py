import numpy as np
import cv2
import sys
import os

def EstimateFundamentalMatrix(image1_points, image2_points):
    CheckFormat(image1_points,image2_points)
    A,T1,T2 = ConstructAMatrix(image1_points,image2_points)

    return EstimateFfromA(A,T1,T2)

# ======================================================================================================
def NormalizePoints(pts):
    # Normalize points to have mean of 0 and mean distance sqrt(2)
    mean = np.mean(pts, axis=0)
    # print(f"me/an: {mean}")
    mean_dist = np.mean(np.linalg.norm(pts - mean, axis=1))
    scale = np.sqrt(2) / mean_dist

    # Construct the normalization matrix
    T = np.array([[scale, 0, -scale * mean[0]],
                  [0, scale, -scale * mean[1]],
                  [0, 0, 1]])
    return T


def CheckFormat(image1_points,image2_points):
    assert len(image1_points) == 8 and len(image2_points)== 8 , 'No. of point correspondences should be exactly 8.'

def ConstructAMatrix(image1_points,image2_points):
    # print(image1_points)
    T1 = NormalizePoints(image1_points)
    T2 = NormalizePoints(image2_points)

    image1_pts_normalized = np.dot(T1, np.vstack((image1_points.T, np.ones(image1_points.shape[0]))))
    image2_pts_normalized = np.dot(T2, np.vstack((image2_points.T, np.ones(image2_points.shape[0]))))
    
    A = np.zeros((image1_points.shape[0], 9))
    # Look Kronecker product
    for i in range(image1_points.shape[0]):
        A[i] = np.kron(image2_pts_normalized[:, i], image1_pts_normalized[:, i])
    return A,T1,T2
        
def EstimateFfromA(A,T1,T2):
    # Cleanup
    U, S, V = np.linalg.svd(np.dot(A.T,A),full_matrices=True)
    F_est = V[-1, :]
    F_est = F_est.reshape(3,3)
    # Enforcing rank 2 for F
    ua,sa,va = np.linalg.svd(F_est,full_matrices=True)
    sa = np.diag(sa)
    sa[2,2] = 0
    F = np.dot(ua,np.dot(sa,va))
    
    F = np.dot(T2.T, np.dot(F, T1))

    # U, S, VT = np.linalg.svd(A,full_matrices=True)
    # F_est = VT[-1, :].reshape((3,3))
    
    # u,s,vT = np.linalg.svd(F_est)
    # s = np.diag(s)
    # s[2,2] = 0
    # F = np.dot(u,np.dot(s,vT))
    return F/F[2,2]



if __name__=='__main__':
    image1_points = np.zeros((8,2))
    image2_points = np.zeros((8,2))
    EstimateFundamentalMatrix(image1_points, image2_points)