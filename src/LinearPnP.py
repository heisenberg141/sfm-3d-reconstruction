import numpy as np
# def removeDuplicates(image1_points_3D):
#     image1_points_unique = set()
#     keep_idx = list()
#     for i in range(len(image1_points_3D)):
#         image1_3D = image1_points_3D[i]
#         if (image1_3D[0], image1_3D[1]) in image1_points_unique:
#             continue
#         else:
#             image1_points_unique.add((image1_3D[0], image1_3D[1]))
#             keep_idx.append(i)
#     return image1_points_3D[keep_idx]
        
def find3DIdxs(image1, image13):
    idxs1 = list()
    idxs2 = list()
    for i in range(len(image1)):
        if image1[i][0] in image13[:,0] and image1[i][1] in image13[:,1]:
            idxs1.append(i)
            idx2 = np.where((image13 == image1[i]).all(axis=1))[0]
            idxs2.append(idx2[0])
    return idxs1, idxs2

def generate2D3Dcorresp(inliers_data_dict, points3D, corresp_image12):
    '''
        DESCRIPTION:
        Given 3D points and their correspondences in 2 images, 
        correspondences in the 3rd and so on images.
    '''
    corresp_image_3D = dict()
    print("sample3D point",points3D[0])
    for image_pair in inliers_data_dict:
        if image_pair != (1,2) and image_pair[0]==1:
            # idx of feature matches between second image and points3D in points3D
            points3D_idx,new_img_idx = find3DIdxs(corresp_image12[:,:2],inliers_data_dict[image_pair][0][:,:2])
            image_3D = np.hstack((inliers_data_dict[image_pair][0][:,2:][new_img_idx],points3D[:,:3][points3D_idx],corresp_image12[:,:2][points3D_idx]))
            corresp_image_3D[image_pair] = image_3D
            print(f"corresp bw {image_pair}: {len(image_3D)}")
    return corresp_image_3D
            
def LinearPnP(camera_matrix,correspondences):
    A = np.empty((0, 12), np.float32)

    for i in range(len(correspondences)):

        # image coordinates
        x, y = correspondences[i][0], correspondences[i][1]

        # normalse the image points
        normalised_pts = np.dot(np.linalg.inv(camera_matrix), np.array([[x], [y], [1]]))
        normalised_pts = normalised_pts/normalised_pts[2]

        # corresp 3d coordinates
        X = correspondences[i][2:]
        X = X.reshape((3, 1))

        # convert to homog
        X = np.append(X, 1)

        zeros = np.zeros((4,))
        
        A_1 = np.hstack((zeros, -X.T, normalised_pts[1]*(X.T)))
        A_2 = np.hstack((X.T, zeros, -normalised_pts[0]*(X.T)))
        A_3 = np.hstack((-normalised_pts[1]*(X.T), normalised_pts[0]*X.T, zeros))


        for a in [A_1, A_2, A_3]:
            A = np.append(A, [a], axis=0)

    A = np.float32(A)
    U, S, VT = np.linalg.svd(A)

    V = VT.T
    # last column of v is the solution i.e the required pose
    pose = V[:, -1]
    pose = pose.reshape((3, 4))

    # extract rot and trans
    R_new = pose[:, 0:3]
    T_new = pose[:, 3]
    R_new = R_new.reshape((3, 3))
    T_new = T_new.reshape((3, 1))

    # impose orthogonality constraint
    U, S, VT = np.linalg.svd(R_new)
    R_new = np.dot(U, VT)

    # check det sign
    if np.linalg.det(R_new) < 0:
        R_new = -R_new
        T_new = -T_new
    C_new = -np.dot(R_new.T, T_new)
    return C_new, R_new

