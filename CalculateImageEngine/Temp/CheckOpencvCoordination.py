import numpy as np
import cv2 as cv

def xyz_to_np_point(x, y, z):
    return np.array([[x, y, z]], dtype=np.float32)

camera_matrix = np.array([
    [2666.66667, 0., 960.0],
    [0., 2250.00000, 540.0],
    [0., 0., 1.]
    ])

points_3d = xyz_to_np_point(0, 0, 10)
rvec = np.zeros((3, 1), np.float32)
tvec = np.zeros((3, 1), np.float32)
dist_coeffs = np.zeros((5, 1), np.float32)

points_2d, _ = cv.projectPoints(points_3d,
                                rvec, tvec,
                                camera_matrix,
                                dist_coeffs)

print(points_2d)