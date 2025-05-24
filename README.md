## Overview
This repo does two tasks: (1) based on feature detection and tracking (from feature-tracking-on-200-images-repository), to compute the fundamental matrix and essential matrix between the given images, and decompose the essential matrix into rotation and translation matrix to plot the trajectory. (2) Triangulate the matching pixels and output a point cloud.

## The implementation performs the following tasks:

1. Feature Detection & Matching  
- SIFT is used for detecting and computing feature descriptors
- Brute-force matcher (BFMatcher with Lowe’s ratio test) is used to match features between consecutive frames

2. Fundamental and Essential Matrix Computation
- Fundamental matrix is estimated using RANSAC
- Essential matrix is computed using camera intrinsics

3. Pose Recovery
- Decomposition of the essential matrix provides rotation and translation between frames
- Global pose is estimated by chaining these relative transformations

4. Trajectory Plotting
- The robot’s trajectory is visualized in 2D by plotting the translation path

5. 3D Point Cloud Generation
- Triangulation is performed on matched keypoints to estimate 3D coordinates
- Points are filtered for visibility and distance from the camera

6. Visualization and Video Output  
- Top panel: RGB image with tracked feature points
- Bottom panel: view of accumulated 3D point cloud and trajectory line
- Frames are saved into a video using OpenCV

## Requirements:
- Python 3.x
- OpenCV
- Matplotlib
- NumPy
- tqdm

You can install using pip:
pip install opencv-python matplotlib numpy tqdm

Run the main script: python code.py

Change the address to the folder containing image frames and path to output on your computer.

