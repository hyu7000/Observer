import cv2
import numpy as np

# 3D 모델 좌표 (Blender 내 가상의 4개 점)
object_points = np.array([
    [0, 0, 0],   # 원점
    [-6, 0, 0],  # X축 끝점 
    [-3, 0, 0],  # X축 중앙점 
    [0, 6, 0],   # Y축 끝점 
    [0, 3, 0],   # Y축 중앙점 
    [0, 0, 6],   # Z축 끝점 
    [0, 0, 3]    # Z축 중앙점 
], dtype=np.float32)

# 감지된 첫 프레임 2D 픽셀 좌표 (렌더링된 영상에서 추출된 값)
image_1_points = np.array([
    [804.0, 484.0],  # (0,0,0)
    [1121.0, 593.0], # (-6,0,0)
    [958.0, 537.0],  # (-3,0,0)
    [596.0, 685.0],  # (0,6,0)
    [705.0, 580.0],  # (0,3,0)
    [790.0, 173.0],  # (0,0,6)
    [798.0, 355.0],  # (0,0,3)
], dtype=np.float32)

# 감지된 두 번째 프레임 2D 픽셀 좌표 (렌더링된 영상에서 추출된 값)
image_2_points = np.array([
    [1121.0, 593.0],   # (-6,0,0)
    [1475.0, 715.0],   # (-12,0,0)
    [1293.0, 653.0],   # (-9,0,0)
    [934.0, 819.0],    # (-6,6,0)
    [1032.0, 700.0],   # (-6,3,0)
    [1136.0, 274.0],   # (-6,0,6)
    [1032.0, 700.0],   # (-6,0,3)
], dtype=np.float32)

# Blender 카메라 내부 행렬 (K)
camera_matrix = np.array([
    [2666.67, 0, 960],
    [0, 1500, 540],
    [0, 0, 1]
], dtype=np.float32)

# 왜곡 계수 (Blender는 디지털 환경이라 기본적으로 왜곡 없음)
dist_coeffs = np.zeros(5)  

# PnP를 이용한 3D 위치 추정 (첫 번째 프레임)
success, rvec, tvec = cv2.solvePnP(object_points, image_1_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
rotation_matrix, _ = cv2.Rodrigues(rvec)

# 카메라의 위치 계산 (월드 좌표계 기준)
camera_position = -np.dot(rotation_matrix.T, tvec)

# 카메라가 바라보는 방향 벡터
camera_direction = np.dot(rotation_matrix.T, np.array([0, 0, 1]))

print(f"📌 PnP 결과 (Frame 1): {success}")
print("회전 행렬 (Rotation Matrix):\n", rotation_matrix)
print("이동 벡터 (Translation Vector):\n", tvec)
print("📌 카메라 위치 (월드 좌표계):\n", camera_position)
print("📌 카메라 바라보는 방향 벡터:\n", camera_direction)

# PnP를 이용한 3D 위치 추정 (두 번째 프레임)
success, rvec, tvec2 = cv2.solvePnP(object_points, image_2_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
rotation_matrix2, _ = cv2.Rodrigues(rvec)

# 카메라의 위치 계산 (월드 좌표계 기준)
camera_position2 = -np.dot(rotation_matrix2.T, tvec2)

# 카메라가 바라보는 방향 벡터
camera_direction2 = np.dot(rotation_matrix2.T, np.array([0, 0, 1]))

print(f"\n📌 PnP 결과 (Frame 2): {success}")
print("회전 행렬 (Rotation Matrix):\n", rotation_matrix2)
print("이동 벡터 (Translation Vector):\n", tvec2)
print("📌 카메라 위치 (월드 좌표계):\n", camera_position2)
print("📌 카메라 바라보는 방향 벡터:\n", camera_direction2)

# 이동 거리 계산
distance = np.linalg.norm(camera_position2 - camera_position)
print(f"\n📌 카메라의 실제 이동 거리: {distance:.2f} cm")
