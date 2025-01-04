from EstimateFundamentalMatrix import *
import numpy as np
import cv2

def GenerateInlierDict(data_dict):
    inlier_dict = dict()
    for corr in data_dict:
        correspondences = np.array(data_dict[corr])
        print(f"Getting Inliers for {corr}")
        inliers, F = GetInlierRANSAC(correspondences[:,:2],correspondences[:,2:],max_iters = 1000)
        print(f"# inliers: {len(inliers)}\n")
        inlier_dict[corr] = [inliers,F]
    return inlier_dict

def GetInlierRANSAC(image1_points, image2_points, max_iters = 1000):
    assert len(image1_points)==len(image2_points)
    i = 0
    n_points = len(image1_points)
    best_F = None
    best_num_inliers = 0
    target_n_inliers = int(len(image1_points)*0.8)
    while(target_n_inliers>best_num_inliers and i <= max_iters):
        idxs = np.random.choice(n_points,8)
        F_curr = EstimateFundamentalMatrix(image1_points[idxs], image2_points[idxs])

        epipolar_distances = compute_epipolar_distance(image1_points, image2_points, F_curr)
        
        inlier_idx = np.where(abs(epipolar_distances) < 0.04)[0]
        num_inliers = len(inlier_idx)
        
        if num_inliers > best_num_inliers:
            best_F = F_curr
            best_inliers_idx = inlier_idx
            best_num_inliers = num_inliers
        i+=1
    
    best_inliers = np.hstack((image1_points[inlier_idx], image2_points[inlier_idx]))
    return best_inliers, best_F
        

# ===================================================================================================

def compute_epipolar_distance(X1, X2, F):
    # Add a column of ones to the points arrays to make them homogeneous
    X1_h = np.hstack((X1, np.ones((X1.shape[0], 1))))
    X2_h = np.hstack((X2, np.ones((X2.shape[0], 1))))

    # Compute epipolar distances
    epipolar_distances = np.abs(np.sum(X2_h * np.dot(F, X1_h.T).T, axis=1))

    return epipolar_distances

if __name__=='__main__':
    image1_points = np.zeros((40,2))
    image2_points = np.zeros((40,2))
    GetInlierRANSAC(image1_points, image2_points, iters = 100)