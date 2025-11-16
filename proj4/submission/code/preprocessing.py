import cv2
import numpy as np
import os
import glob
import viser
import time

def resize_images(source_folder, dest_folder, target_width):
    file_paths = glob.glob(os.path.join(source_folder, '*.jpg'))
    
    for i, file_path in enumerate(file_paths):
        img = cv2.imread(file_path)
        H, W = img.shape[:2]
        
        # maintain aspect ratio
        if W > target_width:
            aspect_ratio = H / W
            new_width = target_width
            new_height = int(new_width * aspect_ratio)
        else:
            new_width = W
            new_height = H

        if W != new_width:
            resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            resized_img = img

        filename = os.path.basename(file_path)
        save_path = os.path.join(dest_folder, filename)
        cv2.imwrite(save_path, resized_img)

def calibrate_camera(folder="calibration_images", size=0.06):
    # Create ArUco dictionary and detector parameters (4x4 tags)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()

    tag_points = np.array([
        [0, 0, 0],
        [size, 0, 0],
        [size, size, 0],
        [0, size, 0]
    ], dtype=np.float32)

    img_points = []
    obj_points = []

    file_paths = glob.glob(f"{folder}/*.jpg")

    for file in file_paths:
        print(file)
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers in an image
        # Returns: corners (list of numpy arrays), ids (numpy array)
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            for corner in corners:
                img_points.append(corner.squeeze())
                obj_points.append(tag_points)
        else: # skip if no tags detected
            pass
    
    if img_points:
        img_size = gray.shape[::-1] # width, height
        ret, cmat, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points,
            img_points,
            img_size,
            None,
            None
        )
        np.save(f'{folder}_camera_matrix.npy', cmat)
        np.save(f'{folder}_dist_coeffs.npy', dist)
        np.savez(f'{folder}_camera_calibration.npz', camera_matrix=cmat, dist_coeffs=dist)

def extract_camera_pose(folder, size=0.06):
    K = np.load(f"{folder}_camera_matrix.npy")
    D = np.load(f"{folder}_dist_coeffs.npy")

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()

    tag_points = np.array([
        [0, 0, 0],
        [size, 0, 0],
        [size, size, 0],
        [0, size, 0]
    ], dtype=np.float32)

    server = viser.ViserServer(share=True)

    file_paths = glob.glob(f"{folder}/*.jpg")
    for i, file in enumerate(file_paths):
        print(file)
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = img.shape[:2]

        # Detect ArUco markers in an image
        # Returns: corners (list of numpy arrays), ids (numpy array)
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)

        # Check if any markers were detected
        if ids is not None:
            img_points = corners[0].squeeze() # (1, 4, 2) -> (4, 2)
            success, rvec, tvec = cv2.solvePnP(
                tag_points,
                img_points,
                K,
                D
            )

            if success:
                # convert to 3d rotation matrix
                R, _ = cv2.Rodrigues(rvec)

                c2w = np.eye(4)
                c2w[:3, :3] = R.T
                c2w[:3, 3] = -R.T @ tvec.flatten()

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                server.add_camera_frustum( # server.scene.add_camera_frustum(
                    f"/cameras/{i}", # give it a name
                    fov=2 * np.arctan2(H / 2, K[0, 0]), # field of view
                    aspect=W / H, # aspect ratio
                    scale=0.02, # scale of the camera frustum change if too small/big
                    wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz, # orientation in quaternion format
                    position=c2w[:3, 3], # position of the camera
                    image=img_rgb # image to visualize
                )
        else:
            pass

    # while True:
    #     time.sleep(0.1)  # Wait to allow visualization to run
    return

def create_nerf_dataset(folder, output_file='nerf_data.npz', size=0.06, train_ratio=0.8, val_ratio=0.15):
    K = np.load(f"{folder}_camera_matrix.npy")
    D = np.load(f"{folder}_dist_coeffs.npy")

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    tag_points = np.array([
        [0, 0, 0],
        [size, 0, 0],
        [size, size, 0],
        [0, size, 0]
    ], dtype=np.float32)

    images = []
    c2ws = []

    file_paths = sorted(
        glob.glob(f"{folder}/*.jpg"),
        key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1])
    )
    for i, file in enumerate(file_paths):
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]

        # Detect ArUco markers in an image
        # Returns: corners (list of numpy arrays), ids (numpy array)
        corners, ids, _ = detector.detectMarkers(gray)

        # Check if any markers were detected
        if ids is not None:
            img_points = corners[0].squeeze() # (1, 4, 2) -> (4, 2)
            success, rvec, tvec = cv2.solvePnP(
                tag_points,
                img_points,
                K,
                D
            )

            if success:
                # convert to 3d rotation matrix
                R, _ = cv2.Rodrigues(rvec)

                c2w = np.eye(4)
                c2w[:3, :3] = R.T
                c2w[:3, 3] = -R.T @ tvec.flatten()

                # alpha=1 keeps all pixels (more black borders), alpha=0 crops maximally
                new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                    K, D, (w, h), alpha=0, newImgSize=(w, h)
                )
                undistorted_img = cv2.undistort(img, K, D, None, new_camera_matrix)

                # Crop to the valid region of interest
                x, y, w_roi, h_roi = roi
                undistorted_img = undistorted_img[y:y+h_roi, x:x+w_roi]

                # Update the principal point to account for the crop offset
                new_camera_matrix[0, 2] -= x  # cx
                new_camera_matrix[1, 2] -= y  # cy

                # if i == 0:
                #     final_camera_matrix = new_camera_matrix.copy()
                if 'final_camera_matrix' not in locals():
                    final_camera_matrix = new_camera_matrix.copy()
                
                undistorted_rgb = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)
              
                images.append(undistorted_rgb)
                c2ws.append(c2w)
        else:
            pass
    
    images = np.array(images, dtype=np.uint8)
    c2ws = np.array(c2ws, dtype=np.float32)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    idx = np.random.permutation(n)

    images_train = images[idx[:n_train]]
    c2ws_train = c2ws[idx[:n_train]]

    images_val = images[idx[n_train:n_train+n_val]]
    c2ws_val = c2ws[idx[n_train:n_train+n_val]]
    
    c2ws_test = c2ws[idx[n_train+n_val:]]

    focal = float(final_camera_matrix[0, 0])

    '''
    images_train: numpy array of shape (N_train, H, W, 3) containing your undistorted training images (0-255 range, will be normalized when loaded)
    c2ws_train: numpy array of shape (N_train, 4, 4) containing the camera-to-world transformation matrices for training images
    images_val: numpy array of shape (N_val, H, W, 3) for validation images
    c2ws_val: numpy array of shape (N_val, 4, 4) for validation camera poses
    c2ws_test: numpy array of shape (N_test, 4, 4) for test camera poses (used for novel view rendering)
    focal: float representing the focal length from your camera intrinsics (assuming fx = fy)
    You can save your dataset using np.savez():
    '''
    np.savez(
        f'nerf/{output_file}',
        images_train=images_train,    
        c2ws_train=c2ws_train,         
        images_val=images_val,    
        c2ws_val=c2ws_val,        
        c2ws_test=c2ws_test,        
        c2ws_all=c2ws,
        focal=focal,                   # float
        K=final_camera_matrix.astype(np.float32)
    )


# resize_images('mofusand_images', 'mofusand_images', 150)
# resize_images('cube_images', 'cube_images', 200)

# calibrate_camera("calibration_images", size=0.06)
# calibrate_camera("mofusand_images", size=0.1025)
# calibrate_camera("cube_images", size=0.1025)

# extract_camera_pose("calibration_images", size=0.06)
# extract_camera_pose("mofusand_images", size=0.1025)
# extract_camera_pose("cube_images", size=0.1025)

# create_nerf_dataset("calibration_images", "calibration_data.npz", size=0.06)
# create_nerf_dataset("mofusand_images", "mofusand_data.npz", size=0.1025)
create_nerf_dataset("cube_images", "cube_data.npz", size=0.1025)

print(np.load("cube_images_camera_matrix.npy"))
print(np.load("cube_images_dist_coeffs.npy"))

