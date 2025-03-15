import numpy as np
import cv2

# === 1. Blender에서 가져온 카메라 내부 행렬 (K) ===
K = np.array([
    [2666.66667, 0, 960],  # fx,  cx
    [0, 2666.00000, 540],  # fy,  cy
    [0, 0, 1]
])

# === 2. Blender에서 가져온 카메라 외부 행렬 (R, t) ===
R_blender = np.array([
    [1.0,  0.0,  0.0],
    [0.0,  1.0,  0.0],
    [0.0,  0.0,  1.0]
])

t_blender = np.array([[0.0], [0.0], [0.0]])  # 3x1 이동 벡터

# === 3. OpenCV 좌표계 변환 (Blender → OpenCV) ===
R_opencv = np.array([
    [1,  0,  0],
    [0,  -1,  0],  # Y → Z
    [0, 0,  1]   # Z → -Y
]) @ R_blender  # 변환 적용

t_opencv = np.array([
    [t_blender[0][0]], 
    [-t_blender[2][0]],  # Z → -Y
    [t_blender[1][0]]    # Y → Z
])

print(f'R_opencv : {R_opencv}')
print(f't_opencv : {t_opencv}')

# === 1. Blender에서 가져온 3D 모델 좌표 ===
object_points = np.array([
#     X   Y   Z
#    [6,  0, -30], # 블렌더 본 좌표(-6,0,-30) x 좌표가 반대
#    [3,  0, -30], # 블렌더 본 좌표(-3,0,-30) x 좌표가 반대
#    [0, -3, -30]  # 블렌더 본 좌표(0,3,-30) x 좌표가 반대   
    [0, -6, -30]  # 블렌더 본 좌표(0,6,-30) x 좌표가 반대   
], dtype=np.float32)

# === 5. 감지된 첫 프레임 2D 픽셀 좌표 (렌더링된 영상에서 추출된 값) ===
image_1_points = np.array([
#    [426.0, 540.0],   # (-6,0,-30) : 블렌더 좌표
#    [692.0, 540.0],   # (-6,0,-30) : 블렌더 좌표
#    [960.0, 272.0],   # (0,3,-30) : 블렌더 좌표
    [960.0, 4.0],   # (0,3,-30) : 블렌더 좌표
], dtype=np.float32)

# === 6. 3D -> 2D 투영 변환 수행 ===
# OpenCV의 projectPoints() 함수 사용 (Rodrigues 변환 제거)
projected_points, _ = cv2.projectPoints(object_points, R_opencv, t_opencv, K, None)

# === 7. 오차 측정 ===
projected_points = projected_points.reshape(-1, 2)  # (7, 2)로 변환
errors = np.linalg.norm(projected_points - image_1_points, axis=1)

# 결과 출력
for i, error in enumerate(errors):
    print(f"Point {i + 1}: 예상 좌표 {projected_points[i]}, 실제 좌표 {image_1_points[i]}, 오차 {error:.2f}")

# === 8. 오차 임계값 확인 ===
threshold = 5.0  # 5픽셀 이하 차이를 허용
if np.all(errors < threshold):
    print("✅ 3D-2D 좌표 매칭이 정확합니다.")
else:
    print("⚠️ 일부 3D-2D 매칭이 부정확합니다.")
