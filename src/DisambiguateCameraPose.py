import numpy as np
from LinearTriangulation import *
from visualization import *

def ComputeCheirality(r3,points3D,c):
    
    x_minus_c = points3D - c
    r3_dot_x_minus_c_T  = np.dot(r3,x_minus_c.T)
    n_z_plus = len(np.where(r3_dot_x_minus_c_T>=0)[0])
    
    return n_z_plus

def DisambiguatePose(R_list, C_list, K, corresp_1_2):
    max_pos_dep = 0
    max_pos_dep_id = 0
    
    for i in range(len(R_list)):
        points3D = LinearTriangulation(np.eye(3), 
                                       np.array([0,0,0]), 
                                       R_list[i], 
                                       C_list[i], 
                                       K, 
                                       corresp_1_2)

        # plot_cameras_and_points_in_xz_plane([0,0,0],C_list[i],R_list[i],points3D)
        
        points3D = points3D[:,0:3]
       
       # Count Positive Depth points for Cam1
        positive_depth = ComputeCheirality(np.array([0,0,1.0]),points3D,np.array([0,0,0]))

        # Count Positive Depth points for Cam2
        r3 = R_list[i][-1]
        positive_depth+=ComputeCheirality(r3,points3D,C_list[i] )
        
        if(positive_depth>max_pos_dep):
            max_pos_dep = positive_depth
            max_pos_dep_id = i
    


    return max_pos_dep_id

