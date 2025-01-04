import numpy as np
import LoadData
from GetInlierRANSAC import *
from visualization import *
import time
import os
from EssentialFromFundamental import *
from ExtractCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonlinearTriangulation import *
from PnPRansac import *

def main():
    data_dir = "../SFM_data/"
    data_dict, images = LoadData.LoadFeatureCorrespondences(data_dir)
    data_dict = LoadData.removeDuplicateCorrespondences(data_dict)
    images = [data_dir + img for img in images]
    images.sort()
    inliers_data_dict = GenerateInlierDict(data_dict) 



    inliers,F = inliers_data_dict[(1,2)]
    plot_inliers_outliers(images[0],images[1],np.array(data_dict[(1,2)]), inliers)

    K =  np.array([ [531.122155322710, 0, 407.192550839899],
                    [0, 531.541737503901, 313.308715048366],
                    [0, 0, 1]
                  ])
    E = EssentialFromFundamental(F,K)

    R_list, C_list = ExtractCameraPose(E)
    
    correct_id = DisambiguatePose(R_list, C_list, K, inliers)
    R,C = R_list[correct_id], C_list[correct_id]
    points3D = LinearTriangulation(np.eye(3),np.array([0,0,0]),R_list[correct_id],C_list[correct_id],K,inliers)
    Camera_RT_List = [[np.eye(3), np.array([0,0,0])]]
    Camera_RT_List.append([R,C])
    plot_cameras_and_points_in_xz_plane_experimental(Camera_RT_List,points3D)
    points3D_optimized = points3D
    points3D_optimized = NonlinearTriangulation(points3D_optimized, inliers,R,C,K)
    plot_cameras_and_points_in_xz_plane_experimental(Camera_RT_List,points3D,points3D_optimized)

    image_3d_corresp = generate2D3Dcorresp(inliers_data_dict,points3D_optimized,inliers)
    print(image_3d_corresp[(1,3)][0])
    for imageid in image_3d_corresp:
        C_, R_ = PnPRansac(image_3d_corresp[imageid][:,:5], K)
        Camera_RT_List.append([R_,C_])
        plot_cameras_and_points_in_xz_plane_experimental(Camera_RT_List, points3D_optimized)
    
    


    

if __name__ =='__main__':
    main()