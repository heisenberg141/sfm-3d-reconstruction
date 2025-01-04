import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import os
import matplotlib.colors as mcolors

def plot_inliers_outliers(img1_dir, img2_dir,corresp_1_2,inliers):
    img1 = cv2.imread(img1_dir)
    img2 = cv2.imread(img2_dir)
    combined_img = np.hstack((img1, img2))
    red = (60,20,220)
    green = (22,225,18)
    
    for point_pair in corresp_1_2:
        # point_pair = point_pair.astype(np.uint8)
        point_pair = np.round(point_pair).astype(np.uint32)
        u1,v1,u2,v2 = point_pair   
        cv2.circle(combined_img, (u1, v1), 2, red, thickness=3)
        u2+=img1.shape[1]
        v2+=img1.shape[2]
        cv2.circle(combined_img, (u2, v2), 2, red, thickness=3)
        # cv2.line(combined_img, (u1,v1), (u2,v2), color, thickness=1)
    color = (0,255,0)
    for point_pair in inliers:
        # point_pair = point_pair.astype(np.uint8)
        point_pair = np.round(point_pair).astype(np.uint32)
        u1,v1,u2,v2 = point_pair   
        cv2.circle(combined_img, (u1, v1), 2, green, thickness=3)
        u2+=img1.shape[1]
        v2+=img1.shape[2]
        cv2.circle(combined_img, (u2, v2), 2, green, thickness=3)
        cv2.line(combined_img, (u1,v1), (u2,v2), green, thickness=2)
    
    plt.imshow(combined_img[:, :, ::-1])
    plt.show()


def VisualizeTriangulation(Points3D, R, C):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot camera centers
    ax.scatter(0, 0, 0, c='r', marker='o', label='Camera Center')
    ax.scatter(C[0],C[1],C[2], c='r', marker='o', label='Camera Center')
    z1_axis = np.array([0,0,1])
    z2_axis = R[:,2]

    # Plot 3D points
    ax.scatter(Points3D[:, 0], Points3D[:, 1], Points3D[:, 2], c='b', marker='^', label='3D Points')
    ax.quiver(0, 0, 0, z1_axis[0], z1_axis[1], z1_axis[2], length=0.5, color='b', label='Z-axis')
    ax.quiver(C[0], C[1], C[2], z2_axis[0], z2_axis[1], z2_axis[2], length=0.5, color='b', label='Z-axis')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot')

    # Add legend
    ax.legend()
    plt.show()

def check_orthonormality(transformed_basis, tol=1e-6):
    # Check orthogonality
    orthogonality = np.allclose(np.dot(transformed_basis, transformed_basis.T), np.eye(3), atol=tol)
    
    # Check normalization
    normalization = np.allclose(np.linalg.norm(transformed_basis, axis=1), 1, atol=tol)
    
    return orthogonality and normalization

def plot_points_in_xz_plane(points):
    # Extract X and Z coordinates
    x_coordinates = points[:, 0]
    z_coordinates = points[:, 2]

    # Plot points in X-Z plane
    plt.scatter(x_coordinates, z_coordinates)
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Points in X-Z Plane')
    plt.grid(True)
    plt.show()

def plot_basis_vectors(point, rotation_matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the given point
    ax.scatter(point[0], point[1], point[2], c='r', marker='o', label='Point')
    ax.scatter(0, 0, 0, c='r', marker='o', label='Point')
    # Transform the unit vectors along X, Y, and Z axes by the rotation matrix
    basis_vectors = np.eye(3)  # Unit vectors along X, Y, Z axes
    transformed_basis = np.dot(rotation_matrix, basis_vectors.T).T
    print(f"Orthonormal? ", check_orthonormality(transformed_basis))
    length_basis = 0.1
    transformed_basis*=length_basis
    basis_vectors*=length_basis
    # Plot the basis vectors originating from the given point
    for vec, color, label in zip(transformed_basis, ['r', 'g', 'b'], ['X-axis', 'Y-axis', 'Z-axis']):
        ax.quiver(point[0], point[1], point[2], vec[0], vec[1], vec[2], length =length_basis, color=color, label=label)
    
    for vec, color, label in zip(basis_vectors, ['r', 'g', 'b'], ['X-axis', 'Y-axis', 'Z-axis']):
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], length =length_basis, color=color, label=label)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Basis Vectors from Rotation Matrix')

    # Add legend
    ax.legend()

    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])

    # Show plot
    plt.show()

def plot_cameras_and_points_in_xz_plane(cam1_position, cam2_position, cam2_rotation_matrix, points, points_optim = list()):
    # Plot camera positions and Z-axis directions
    cam2_position = cam2_position.reshape(3,)
    plt.scatter(cam1_position[0], cam1_position[2], color='b', label='Camera 1 Position')
    plt.scatter(cam2_position[0], cam2_position[2], color='r', label='Camera 2 Position')

    # Direction of Z-axis for Camera 1
    cam1_z_axis = np.array([0, 0, 1])
    cam1_z_axis = cam1_position + cam1_z_axis
    plt.plot([cam1_position[0], cam1_z_axis[0]], [cam1_position[2], cam1_z_axis[2]], color='b', label='Camera 1 Z-axis')

    # Direction of Z-axis for Camera 2
    cam2_z_axis = np.dot(cam2_rotation_matrix, np.array([0, 0, 1]))
    cam2_z_axis = cam2_position + cam2_z_axis
    plt.plot([cam2_position[0], cam2_z_axis[0]], [cam2_position[2], cam2_z_axis[2]], color='r', label='Camera 2 Z-axis')

    # Plot 3D points
    x_coordinates = points[:, 0]
    z_coordinates = points[:, 2]
    plt.scatter(x_coordinates, z_coordinates, c='g', label='3D Points',s=10)
    if len(points_optim)>0:
        x_coordinates = points_optim[:, 0]
        z_coordinates = points_optim[:, 2]
        plt.scatter(x_coordinates, z_coordinates, c='lightblue', label='Optimized 3D Points',s=2)
    plt.xlim(-20, 20)
    plt.ylim(-5, 20)

    # Plot settings
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Cameras, Z-axis Directions, and 3D Points in XZ Plane')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_cameras_and_points_in_xz_plane_experimental(camera_R_T, points, points_optim = list()):
    
    colors = common_colors = [
    'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 
    'black', 'white', 'gray', 'orange', 'purple', 'brown', 
    'pink', 'lime', 'olive', 'navy', 'teal', 'maroon'
]
    for i,transformation in enumerate(camera_R_T):
        camera_position = transformation[1]
        camera_position = camera_position.reshape(3,)
        rotation_matrix = transformation[0]
        cam_z_axis = camera_position + np.dot(rotation_matrix, np.array([0, 0, 1])) 
        
        plt.scatter(camera_position[0], camera_position[2], color = colors[i],label=f'Camera {i+1} Z-axis')
        plt.plot([camera_position[0], cam_z_axis[0]], [camera_position[2], cam_z_axis[2]], color=colors[i])

    x_coordinates = points[:, 0]
    z_coordinates = points[:, 2]
    plt.scatter(x_coordinates, z_coordinates, c='black', label='3D Points',s=10)
    if len(points_optim)>0:
        x_coordinates = points_optim[:, 0]
        z_coordinates = points_optim[:, 2]
        plt.scatter(x_coordinates, z_coordinates, c='lightblue', label='Optimized 3D Points',s=2)
    
    plt.xlim(-15, 15)
    plt.ylim(-2, 25)

    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Cameras with Directions and 3D Points in XZ Plane')
    plt.legend()
    plt.grid(True)
    plt.show()
