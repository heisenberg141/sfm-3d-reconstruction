# Structure from Motion (SfM) with Python and Numpy  

## Overview  
This project implements a Structure from Motion (SfM) pipeline using Python to reconstruct 3D scenes from 2D images. SfM is a computer vision technique used for 3D reconstruction and is widely applied in robotics, AR/VR, and mapping.  

## Pipeline  

The Structure from Motion (SfM) pipeline consists of several stages, each responsible for processing input images and producing intermediate results that contribute to the final 3D reconstruction. Below is a detailed explanation of each step, accompanied by visual examples:

### 1. **Fundamental Matrix Estimation**  
- The pipeline begins by detecting and matching keypoints between image pairs.  
- Using robust estimation techniques like **RANSAC**, the **Fundamental Matrix (F)** is computed, which encodes the epipolar geometry between two images. This step ensures that only inlier matches are used for subsequent calculations.  

![F Matrix](./media/FundamentalMatrix.png)  

### 2. **RANSAC Inliers for Feature Matching**  
- After estimating the Fundamental Matrix, outliers are rejected, leaving only high-confidence feature matches. This process is crucial for stable and accurate reconstruction.  

![RANSAC Inliers](./media/Fundamentalmatrixinliers.png)  

### 3. **Essential Matrix Decomposition**  
- Using the Fundamental Matrix and intrinsic camera parameters, the **Essential Matrix (E)** is computed.  
- The Essential Matrix is decomposed into rotation (R) and translation (t) components to estimate the relative pose between cameras.  

### 4. **Triangulation**  
- Given the relative camera poses and matched feature points, 3D points are reconstructed via **triangulation**.  
- Triangulation uses the epipolar geometry and camera projection matrices to estimate the 3D location of points in the scene.  

![Camera pose estimation and triangulation](./media/Cameraposewithtriangulated3Dpoints.png)  

### 5. **PnP (Perspective-n-Point)**  
- After reconstructing some initial 3D points and camera poses, the pipeline incorporates additional images.  
- Using the **PnP algorithm**, new camera poses are registered by aligning 2D keypoints from the new image with previously reconstructed 3D points.  
- This step incrementally expands the reconstruction.  

![PnP with Multiple Cameras](./media/MultipleCamerasWithPoints.png)  

### 6. **Bundle Adjustment**  
- **Bundle Adjustment** refines the entire structure by simultaneously optimizing:  
  - Camera intrinsic and extrinsic parameters (pose and focal length).  
  - 3D point coordinates.  
- The goal is to minimize the overall **reprojection error**, ensuring that the reconstructed 3D points align well with the observed 2D points in the images.  

### 7. **Visualization**  
- The final output is a 3D point cloud representing the reconstructed scene.  
- The pipeline supports visualizing intermediate and final results, such as camera positions and dense or sparse point clouds.  

---

### Pipeline Flow Diagram  

To provide a clear overview, here’s a step-by-step flow of the pipeline:

1. **Feature Detection & Matching**  
2. **Fundamental Matrix Estimation** (Epipolar Geometry)  
3. **Essential Matrix Decomposition** (Camera Pose)  
4. **Triangulation**

## Dependencies  
- Python 3.8 or higher   
- NumPy  
- Matplotlib  
- SciPy  

Install dependencies using:  
```bash  
pip install opencv-python numpy matplotlib scipy  
