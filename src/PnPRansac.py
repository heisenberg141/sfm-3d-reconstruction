import numpy as np
import cv2
from LinearPnP import *

def reprojectionError(corresp2D3D, C, R, K):
    transformation_mat = np.hstack((R, -np.dot(R,C)))
    projection_mat = np.dot(K,transformation_mat)
    world_points = corresp2D3D[:,2:]
    ones = np.ones((world_points.shape[0], 1))
    world_points_homo = np.hstack((world_points,ones))

    # print("Projection Matrix: ", projection_mat.shape)
    reprojected_points = np.dot(projection_mat,world_points_homo.T)
    image_points = (reprojected_points/reprojected_points[2]).T [:,0:2]
    reprojection_error = np.sqrt(np.linalg.norm(image_points - corresp2D3D[:,0:2], axis = 1))
    # print(distance[0])
    # print("Reprojected_points:",reprojected_points.shape)

    # P_Matrix = np.dot()
    return reprojection_error


def PnPRansac(corresp2D3D, K, max_iters = 1000):
    i = 0
    C_optim,R_optim = LinearPnP(K, corresp2D3D[:6,:])
    max_inliers = 0

    while(i < max_iters):
        random_indices = np.random.choice(len(corresp2D3D),6)
        C,R = LinearPnP(K, corresp2D3D[random_indices])
        reproj = reprojectionError(corresp2D3D, C, R, K)
        curr_inliers = len(np.where(abs(reproj)<5)[0])
        if curr_inliers>max_inliers:
            max_inliers = curr_inliers
            C_optim, R_optim = C, R
        i+=1
    print("MaxInliers: ", max_inliers/len(corresp2D3D)*100, "%")
    print(C_optim, R_optim)
    return C_optim,R_optim