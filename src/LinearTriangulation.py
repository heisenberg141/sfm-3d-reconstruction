import numpy as np

def LinearTriangulation(R1, C1, R2, C2, K, correspondences):
    P1 = np.dot(K, np.dot(R1, np.hstack((np.eye(3),-C1.reshape(3,1)))))
    P2 = np.dot(K, np.dot(R2, np.hstack((np.eye(3),-C2.reshape(3,1)))))

    img1_points = correspondences[:,0:2]
    img2_points = correspondences[:,2:]
    p1, p2, p3 = P1
    p1_dash, p2_dash, p3_dash = P2
    
    points3D = list()
    
    for i in range(len(img1_points)):
        A = list()
        x,y  = img1_points[i]
        x_dash, y_dash = img2_points[i]
        # print(np.dot(y,p3.T)-p2.T)
       
        A.append(np.dot(y,p3.T)-p2.T)
        A.append(p1.T - np.dot(x, p3.T))
        A.append(np.dot(y_dash,p3_dash.T)- p2_dash.T)
        A.append(p1_dash.T-np.dot(x_dash, p3_dash.T))
        
        A = np.array(A)

        _,_,VT = np.linalg.svd(A, full_matrices=True)
        position_3d = VT[-1,:]
        position_3d/=position_3d[-1]
        # print(position_3d)
        points3D.append(position_3d)
    return np.array(points3D)

    




    






