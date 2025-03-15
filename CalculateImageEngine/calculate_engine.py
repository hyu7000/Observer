import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass

@dataclass
class DeltaMotion:
    distance:float
    direction:list
    angle:float
    angle_x:float
    angle_y:float
    angle_z:float

class CalculateImageEngine:
    def __init__(self, edgeLength:int, camera_k, dist_coeffs=None):
        # 3D 모델 좌표
        self.obj_points = np.array([
            [0,           0,            0         ], # 원점
            [0,           0,            edgeLength], # Z축 끝점 
            [0,           edgeLength,   0         ], # Y축 끝점 
            [edgeLength,  0,            0         ], # X축 끝점 
        ], dtype=np.float32)

        # Blender 카메라 내부 행렬 (K)
        self.camera_k = camera_k

        # 왜곡 계수
        if dist_coeffs is None:
            self.dist_coeffs = np.zeros(5)
        else:
            self.dist_coeffs = dist_coeffs

        self.delta_motions = []
    
    def __get_trace_data(self, image1_points, image2_points) -> DeltaMotion:
        """solvePnP 결과를 객체 좌표계 기준으로 변환하여 이동 및 회전 계산"""

        # ✅ 첫 번째 프레임: 객체 → 카메라 좌표 변환 행렬 구하기 (더 정밀한 PnP 알고리즘 사용)
        success, rvec1, tvec1 = cv2.solvePnP(
            self.obj_points, image1_points, self.camera_k, self.dist_coeffs, flags=cv2.SOLVEPNP_AP3P
        )
        R_cam1, _ = cv2.Rodrigues(rvec1)
        
        # ✅ 두 번째 프레임
        success, rvec2, tvec2 = cv2.solvePnP(
            self.obj_points, image2_points, self.camera_k, self.dist_coeffs, flags=cv2.SOLVEPNP_AP3P
        )
        R_cam2, _ = cv2.Rodrigues(rvec2)

        # ✅ 카메라 좌표계 기준 이동량 및 회전량
        tvec_delta_cam = tvec2 - tvec1
        R_delta = np.dot(R_cam2, R_cam1.T)
        
        # ✅ 회전 행렬을 안정적으로 변환 (정규화)
        U, _, Vt = np.linalg.svd(R_delta)  # 특이값 분해(SVD)로 정규화
        R_delta = np.dot(U, Vt)

        rvec_delta_cam, _ = cv2.Rodrigues(R_delta)

        # ✅ 객체 좌표계로 변환: R_obj = R_cam.T
        R_obj = R.from_matrix(R_cam1).inv().as_matrix()  # scipy를 활용한 더 정확한 역변환
        
        # ✅ 이동 벡터를 객체 좌표계로 변환
        tvec_delta_obj = np.dot(R_obj, tvec_delta_cam)

        # ✅ 회전 벡터를 객체 좌표계로 변환
        rvec_delta_obj = np.dot(R_obj, rvec_delta_cam)

        # ✅ 회전량 계산
        theta_rad = np.linalg.norm(rvec_delta_obj)
        theta_deg = np.degrees(theta_rad)

        # ✅ 이동 방향 (객체 좌표계 기준)
        direction_vector_obj = tvec_delta_obj / np.linalg.norm(tvec_delta_obj) if np.linalg.norm(tvec_delta_obj) > 1e-6 else np.array([0, 0, 0])

        # ✅ 각 축(X, Y, Z) 별 회전량
        rotation_x, rotation_y, rotation_z = np.degrees(rvec_delta_obj.flatten())

        deltaMotion = DeltaMotion(
            np.linalg.norm(tvec_delta_obj),
            direction_vector_obj.flatten().tolist(),
            theta_deg,
            rotation_x,
            rotation_y,
            rotation_z
        )

        return deltaMotion
    
    def calculate_images(self, image_points:list):        
        # clear
        self.delta_motions = [] 

        for index, image in enumerate(image_points):
            if index + 1 >= len(image_points):
                break

            deltaMotion = self.__get_trace_data(image, image_points[index+1])
            self.delta_motions.append(deltaMotion)

    def get_distance(self):
        if len(self.delta_motions) == 0:
            return 0
        
        distance = 0
        for delta_motion in self.delta_motions:
            distance += delta_motion.distance
        
        return distance

    
if __name__ == "__main__":

    import time

    # Blender 카메라 내부 행렬 (K)
    camera_matrix = np.array([
        [2666.67, 0, 960],
        [0, 2666.67, 540],
        [0, 0, 1]
    ], dtype=np.float32)

    start = time.perf_counter()

    observer = CalculateImageEngine(6, camera_matrix)

    # 시작 좌표
    # 감지된 첫 프레임 2D 픽셀 좌표 (렌더링된 영상에서 추출된 값)
    image1_points = np.array([
        [783.0, 503.0],   # (0,0,0)
        [545.0, 720.0],   # (6,0,0)
        [1107.0, 636.0],  # (0,6,0)
        [766.0, 183.0],   # (0,0,6)
    ], dtype=np.float32)    

    # 이동 (Z : 2m)
    image4_points = np.array([
        [778.0, 403.0],  # (0,0,0)
        [531.0, 617.0],   # (6,0,0)
        [1112.0, 534.0],  # (0,6,0)
        [759.0, 62.0],  # (0,0,6)
    ], dtype=np.float32)

    # 이동 (Y : 2m, 상대좌표)
    # image5_points = np.array([
    #     [885.0, 445.0],  # (0,0,0)
    #     [644.0, 668.0],   # (6,0,0)
    #     [1233.0, 581.0],  # (0,6,0)
    #     [877.0, 100.0],  # (0,0,6)
    # ], dtype=np.float32)

    # 회전 + 이동 (Y : 2m, Z : -45도, 상대좌표)
    image5_points = np.array([
        [885.0, 445.0],  # (0,0,0)
        [872.0, 708.0],   # (6,0,0)
        [1261.0, 394.0],  # (0,6,0)
        [877.0, 100.0],  # (0,0,6)
    ], dtype=np.float32)

    image_points = [image1_points, image4_points, image5_points]
    observer.calculate_images(image_points)
    print(f'distance : {observer.get_distance()}')

    end = time.perf_counter()
    
    print(f"실행 시간: {end - start:.6f} 초")
    print('\n')