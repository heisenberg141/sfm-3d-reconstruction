import numpy as np
import scipy
from visualization import *



def reprojection_error(gt, P1, P2, img_point):
    reproj_img1 = np.dot(P1,gt.T).T
    reproj_img1/=reproj_img1[-1]
    reproj_img1 = reproj_img1[:2]
    reproj_img2 = np.dot(P2,gt.T).T
    reproj_img2/=reproj_img2[-1]
    reproj_img2 = reproj_img2[:2]
    
    ssd = np.sum((reproj_img1 - img_point[:2]) ** 2)
    ssd += np.sum((reproj_img2 - img_point[2:]) ** 2)
    
    return ssd

def NonlinearTriangulation(points3D, inliers,R2,C2,K):
    points3D_optimized = list()
    R1 = np.eye(3)
    C1 = np.array([0,0,0])
    
    P1 = np.dot(K, np.dot(R1, np.hstack((np.eye(3),-C1.reshape((3,1))))))
    P2 = np.dot(K, np.dot(R2, np.hstack((np.eye(3),-C2.reshape((3,1))))))
    
    SE_point3d = 0
    SE_optimized = 0 
    
    for i in range(len(points3D)):
        # reprojection_error(reprojected_img1[i], reprojected_img2[i],img1_pts[i], img2_pt[i])
        SE_point3d+=reprojection_error(points3D[i],P1,P2, inliers[i])
        opt = scipy.optimize.least_squares(reprojection_error, points3D[i], args = [P1, P2, inliers[i]])
        SE_optimized+=reprojection_error(opt.x ,P1,P2, inliers[i])
        points3D_optimized.append(opt.x)
    
    MSE_3D = SE_point3d/len(points3D)
    MSE_optim = SE_optimized/len(points3D)
    
    print(f"Reduced the reprojection error by {(MSE_3D-MSE_optim)/MSE_3D*100}% ")

    return np.array(points3D_optimized)
    

     