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

    DEBUG_MODE = False

    def __init__(self, edgeLength:int, camera_k, dist_coeffs=None):
        # 3D 모델 좌표
        self.obj_points = np.array([
            [0,           0,            0         ], # 원점
            [edgeLength,  0,            0         ], # X축 끝점 
            [0,           edgeLength,   0         ], # Y축 끝점 
            [0,           0,            edgeLength], # Z축 끝점 
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
        R_obj1 = R_cam1.T
        t_obj1 = -R_obj1 @ tvec1

        if CalculateImageEngine.DEBUG_MODE:
            print("📷 Frame 1")
            print(f"rvec1: {rvec1.flatten()}")
            print(f"tvec1: {tvec1.flatten()}")
            print(f"R_obj1:\n{R_obj1}")
            print(f"t_obj1: {t_obj1.flatten()}\n")

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

            if CalculateImageEngine.DEBUG_MODE:
                print(f'deltaMotion : {deltaMotion}')
                self.debug_pose_estimation(
                    self.obj_points,
                    image,
                    self.camera_k,
                    self.dist_coeffs,
                    method=cv2.SOLVEPNP_AP3P
                )

    def get_distance(self):
        if len(self.delta_motions) == 0:
            return 0
        
        distance = 0
        for delta_motion in self.delta_motions:
            distance += delta_motion.distance
        
        return distance
    
    def get_angle(self):
        if len(self.delta_motions) == 0:
            return 0
        
        angle = 0
        for delta_motion in self.delta_motions:
            angle += delta_motion.angle
        
        return angle


    # 디버깅용 함수
    def debug_pose_estimation(self, obj_points, img_points, camera_k, dist_coeffs, method=cv2.SOLVEPNP_ITERATIVE):        
        import numpy as np
        import cv2
        """
        solvePnP 디버깅용 유틸 함수
        - 입력: obj_points (3D), img_points (2D), 카메라 내파라미터
        - 출력: 추정된 rvec, tvec, R, 재투영 오차 등
        """
        assert obj_points.shape[0] == img_points.shape[0], "3D-2D 점 개수가 일치하지 않음"
        obj_points = obj_points.astype(np.float32)
        img_points = img_points.astype(np.float32)

        # SolvePnP
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_k, dist_coeffs, flags=method)
        if not success:
            print("❌ solvePnP 실패")
            return None

        R, _ = cv2.Rodrigues(rvec)

        # 재투영
        projected_points, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_k, dist_coeffs)
        projected_points = projected_points.reshape(-1, 2)

        # 오차 계산
        error = np.linalg.norm(projected_points - img_points, axis=1)
        mean_error = np.mean(error)

        print("📌 rvec:", rvec.flatten())
        print("📌 tvec:", tvec.flatten())
        print("📌 재투영 평균 오차:", mean_error)
        for i, (actual, proj, err) in enumerate(zip(img_points, projected_points, error)):
            print(f"  ▸ Point {i}: 실제 = {actual}, 예측 = {proj}, 오차 = {err:.2f}")

        return {
            'rvec': rvec,
            'tvec': tvec,
            'R': R,
            'projected_points': projected_points,
            'error_per_point': error,
            'mean_error': mean_error
        }
    
if __name__ == "__main__":

    import time

    # Blender 카메라 내부 행렬 (K)
    camera_matrix = np.array([
        [2666.67, 0, 960],
        [0, 2666.67, 540],
        [0, 0, 1]
    ], dtype=np.float32)

    # 카메라 좌표
    # X : 26.675mm, Y : 20.272mm, Z : 26.779mm

    start = time.perf_counter()

    observer = CalculateImageEngine(6, camera_matrix)

    # 시작 좌표
    # 감지된 첫 프레임 2D 픽셀 좌표 (렌더링된 영상에서 추출된 값)
    image1_points = np.array([
        [783.0, 503.0],   # (0,0,0)
        [981.0, 322.0],   # (6,0,0)
        [766.0, 183.0],   # (0,0,6)
        [1107.0, 636.0],  # (0,6,0)
    ], dtype=np.float32)    

    # 회전 (Z : -45도)
    image6_points = np.array([
        [783.0,  503.0],
        [1144.0, 453.0],
        [766.0,  183.0],
        [854.0,  758.0],
    ], dtype=np.float32)

    image_points = [image1_points, image6_points]
    # image_points = [image1_points, image4_points]
    # image_points = [image1_points, image5_points]
    observer.calculate_images(image_points)
    print(f'distance : {observer.get_distance()}')
    print(f'angle : {observer.get_angle()}')

    end = time.perf_counter()
    
    print(f"실행 시간: {end - start:.6f} 초")
    print('\n')