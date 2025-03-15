import cv2
import numpy as np

# 3D 모델 좌표 (Blender 내 가상의 점)
object_points = np.array([
    [0, 0, 0],   # 원점
    [-6, 0, 0],  # X축 끝점 
    [0, 6, 0],   # Y축 끝점 
    [0, 0, 6],   # Z축 끝점 
], dtype=np.float32)

# 감지된 첫 프레임 2D 픽셀 좌표 (렌더링된 영상에서 추출된 값)
image_points = np.array([
    [804.0, 484.0],   # (0,0,0)
    [1121.0, 593.0],  # (-6,0,0)
    [596.0, 685.0],   # (0,6,0)
    [790.0, 173.0],   # (0,0,6)
], dtype=np.float32)

# Blender 카메라 내부 행렬 (K)
camera_matrix = np.array([
    [2666.67, 0, 960],
    [0, 2666.67, 540],
    [0, 0, 1]
], dtype=np.float32)

# 왜곡 계수 (Blender는 디지털 환경이라 기본적으로 왜곡 없음)
dist_coeffs = np.zeros(5)  

# PnP를 이용한 3D 위치 추정 (첫 번째 프레임)
success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_AP3P)
rotation_matrix, _ = cv2.Rodrigues(rvec)

# 카메라의 위치 계산 (월드 좌표계 기준)
camera_position = -np.dot(rotation_matrix.T, tvec)

# 카메라가 바라보는 방향 벡터
camera_direction = np.dot(rotation_matrix.T, np.array([0, 0, 1]))

# 너무 작은 수 제거 (1e-10 이하 값은 0으로 변환)
def clean_values(arr, precision=6):
    arr = np.where(np.abs(arr) < 1e-10, 0, arr)  # 작은 값 0 처리
    return np.round(arr, precision)  # 소수점 precision자리까지만 출력

# 최적화된 결과 출력
print(f"📌 PnP 결과 (Frame 1): {success}")
print(f"rvec : {clean_values(rvec)}")
print("회전 행렬 (Rotation Matrix):\n", clean_values(rotation_matrix))
print("이동 벡터 (Translation Vector):\n", clean_values(tvec))
print("📌 카메라 위치 (OpenCV 기준):\n", clean_values(camera_position))
print("📌 카메라 바라보는 방향 벡터 (OpenCV 기준):\n", clean_values(camera_direction))
