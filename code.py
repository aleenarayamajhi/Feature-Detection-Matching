import cv2 as cv
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for headless environments
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

#camera Intrinsic Matrix (provided in the assignment)
K = np.array([
    [707.0493, 0, 604.0814],
    [0, 707.0493, 180.5066],
    [0, 0, 1]
])

#load all image paths from the specified folder
def load_images(image_folder):
    return sorted(glob(os.path.join(image_folder, "*.png")))

#detect and match feature points using SIFT and BFMatcher
def get_feature_points(img1, img2):
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    pts1, pts2 = [], []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:  #Lowe's ratio test
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    return np.array(pts1, dtype=np.float32), np.array(pts2, dtype=np.float32)

#triangulate 3D points from 2D correspondences between two views
def triangulate(R, t, pts1, pts2):
    proj1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  #projection for first camera
    proj2 = K @ np.hstack((R, t.reshape(3, 1)))           #projection for second camera
    pts4d = cv.triangulatePoints(proj1, proj2, pts1.T, pts2.T)
    pts4d /= pts4d[3]  #convert from homogeneous to 3D
    return pts4d[:3].T  #Nx3

#create an image showing the point cloud and trajectory 
def draw_point_cloud(points, trajectory, image_size=(1600, 800)):
    canvas = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)  

    scale = 5  # Zoom factor
    offset_x = image_size[0] // 2
    offset_y = image_size[1] - 100  

    #plot all 3D points in yellow
    for pt in points:
        x, z = int(pt[0] * scale + offset_x), int(-pt[2] * scale + offset_y)
        if 0 <= x < image_size[0] and 0 <= z < image_size[1]:
            cv.circle(canvas, (x, z), 2, (0, 255, 255), -1)

    #Draw trajectory in red
    for i in range(1, len(trajectory)):
        x1, z1 = int(trajectory[i-1][0] * scale + offset_x), int(-trajectory[i-1][2] * scale + offset_y)
        x2, z2 = int(trajectory[i][0] * scale + offset_x), int(-trajectory[i][2] * scale + offset_y)
        if all(0 <= val < image_size[0] for val in [x1, x2]) and all(0 <= val < image_size[1] for val in [z1, z2]):
            cv.line(canvas, (x1, z1), (x2, z2), (0, 0, 255), 2)  #red line

    return canvas

def main():
    image_folder = r"C:\path\to\images_200"
    output_video = "video.mp4"
    image_paths = load_images(image_folder)

    pose = np.eye(4)  #initial pose (identity matrix)
    trajectory = [pose[:3, 3].copy()]  #start trajectory with initial position
    all_points = []  #store all 3D points
    frames = []      #store frames for video

    for i in tqdm(range(len(image_paths) - 1)):
        #read grayscale for feature matching and RGB for display
        img1_gray = cv.imread(image_paths[i], cv.IMREAD_GRAYSCALE)
        img2_gray = cv.imread(image_paths[i+1], cv.IMREAD_GRAYSCALE)
        img1_rgb = cv.imread(image_paths[i])
        img1_rgb = cv.cvtColor(img1_rgb, cv.COLOR_BGR2RGB)

        #feature matching
        pts1, pts2 = get_feature_points(img1_gray, img2_gray)
        if len(pts1) < 8:
            continue  #skip if not enough points

        #estimate essential matrix and recover pose
        E, mask = cv.findEssentialMat(pts1, pts2, K, method=cv.RANSAC, threshold=1.0)
        _, R, t, _ = cv.recoverPose(E, pts1, pts2, K)

        #update global pose
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.ravel()
        pose = pose @ np.linalg.inv(T)  #chain transformations
        trajectory.append(pose[:3, 3].copy())

        #triangulate and transform points to world frame
        pts3d = triangulate(R, t, pts1, pts2)
        pts3d_world = (pose[:3, :3] @ pts3d.T).T + pose[:3, 3]

        #filter points by distance and depth
        center = pose[:3, 3]
        dists = np.linalg.norm(pts3d_world - center, axis=1)
        mask = (dists < 30) & (pts3d_world[:, 2] > 0)
        pts3d_world = pts3d_world[mask]
        all_points.append(pts3d_world)

        #top panel with image and keypoints
        img_disp = img1_rgb.copy()
        for pt in pts1:
            cv.circle(img_disp, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
        top = cv.resize(img_disp, (1280, 512))

        #bottom panel with point cloud + trajectory
        cloud = np.vstack(all_points)
        bottom = draw_point_cloud(cloud, trajectory)
        bottom = cv.resize(bottom, (1280, 512))

        #combine top and bottom
        combined = np.vstack((top, bottom))
        frames.append(combined)

    #write all frames into a video
    out = cv.VideoWriter(output_video, cv.VideoWriter_fourcc(*'mp4v'), 20,
                         (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"video saved to: {output_video}")

if __name__ == "__main__":
    main()
