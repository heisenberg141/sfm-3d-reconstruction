import numpy as np
from visualization import *


def ExtractCameraPose(E):
    U,S,VT = np.linalg.svd(E, full_matrices=True)

    # cam_center = U[:,-1]
    # print(cam_center)
    C = U[:,-1]
    camera_centers = [C, -C, C, -C]
    
    W = np.array([[0, -1.0, 0],[1.0, 0, 0],[0, 0, 1.0]])
    R1 = np.dot(U, np.dot(W,VT))
    R2 = np.dot(U, np.dot(W.T,VT))
    rotation_matrices = [R1, R1, R2, R2]

    for i in range(4):
        if np.linalg.det(rotation_matrices[i])< 0:
            rotation_matrices[i] = -rotation_matrices[i]
            camera_centers[i] = -camera_centers[i]
        
            
    
    return rotation_matrices, camera_centers
