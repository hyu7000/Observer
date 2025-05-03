import cv2

from CaptureRefCube.capture_ref import RecordModule
from CalculateImageEngine.calculate_engine import CalculateImageEngine
from DetectRefCube.detect_ref import DetectModule

class Observer:

    COUNT = 0
    COUNT_AVG_FRAME = 15

    def __init__(self, edge_length:int, camera_k):
        self.record_module   = RecordModule()
        self.cal_imag_engine = CalculateImageEngine(edge_length, camera_k)

        self.is_active_camera = False

        self.is_observing_state = False
        self.record_file_path = f"temp_{Observer.COUNT}.mp4"

    def get_camera_names(self):
        return self.record_module.get_camera_names()
    
    def activate_camera(self, camera_name) -> bool:
        if self.record_module.activate_camera(camera_name):
            self.is_active_camera = True
        return self.is_active_camera
    
    def start_observe(self) -> bool:
        if not self.is_active_camera:
            return False

        if self.record_module.start_record(self.record_file_path):
            self.is_observing_state = True
            Observer.COUNT += 1

        return self.is_observing_state
    
    def stop_observe(self) -> bool:
        if not self.is_active_camera:
            return False

        if not self.is_observing_state:
            return False
        
        if self.record_module.stop_record():
            self.is_observing_state = False

            self.__parse()

        return (self.is_observing_state == False)
    
    def __parse(self):
        detect_module = DetectModule(self.record_module.abs_output_path)

        key_points = detect_module.get_key_points()
        # Use 'ascontiguousarray' because OpenCV requires contiguous memory
        filtered_key_points = [np.ascontiguousarray(kp[:, :-1], dtype=np.float32) for kp in key_points]  # 각 NumPy 배열에서 마지막 열 제거 

        self.cal_imag_engine.calculate_images(filtered_key_points)
    
    def __get_video_fps(self, path:str):
        cap = cv2.VideoCapture(path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        return fps
    
    def __get_avg_keypoint(self, keypoint_list, start_frame: int, end_frame: int):
        """
        keypoint 리스트에서 특정 프레임 구간(n~m)의 평균 좌표 계산

        Args:
            keypoint_list (List[np.ndarray]): (N, 3) 형태의 keypoints를 담은 리스트
            start_frame (int): 시작 프레임 인덱스 (inclusive)
            end_frame (int): 끝 프레임 인덱스 (inclusive)

        Returns:
            np.ndarray | None: (N, 3) 형태의 평균 keypoint 배열 (없으면 None)
        """
        sublist = keypoint_list[start_frame:end_frame+1]
        if not sublist:
            return None

        stacked = np.stack(sublist, axis=0)  # shape: (T, N, 3)
        return np.mean(stacked, axis=0)      # shape: (N, 3)

    def __correct_keypoints_sequence(self, image_points_list, threshold=1, n=1):
        """
        image_points_list: List of np.array, each of shape (4, 2)
        threshold: distance threshold for correction
        n: number of frames before and after to average from (default=1)
        Returns: corrected list of np.array
        """
        corrected_list = []

        total_frames = len(image_points_list)

        for idx in range(total_frames):
            if idx < n or idx >= total_frames - n:
                # 처음과 마지막 n개는 보정하지 않고 그대로 사용
                corrected_list.append(image_points_list[idx])
                continue

            # 평균 프레임 계산 (앞뒤 n개 평균)
            surrounding_points = [image_points_list[i] for i in range(idx - n, idx + n + 1) if i != idx]
            expected_points = sum(surrounding_points) / len(surrounding_points)

            curr = image_points_list[idx]
            corrected = curr.copy()

            for i in range(4):  # keypoint 0~3
                error = np.linalg.norm(curr[i] - expected_points[i])
                if error > threshold:
                    corrected[i] = expected_points[i]  # 보정

            corrected_list.append(corrected)

        return corrected_list

    def parse_video(self, path:str):
        self.fps = self.__get_video_fps(path)

        detect_module = DetectModule(path)

        captured_key_points = detect_module.get_key_points()
        # Use 'ascontiguousarray' because OpenCV requires contiguous memory
        filtered_key_points = [np.ascontiguousarray(kp[:, :-1], dtype=np.float32) for kp in captured_key_points]  # 각 NumPy 배열에서 마지막 열 제거 
        self.key_points = self.__correct_keypoints_sequence(filtered_key_points)

        self.cal_imag_engine.calculate_images(self.key_points)

    def get_distance(self):
        return self.cal_imag_engine.get_distance()
    
    def get_displacement(self):
        if self.key_points is None:
            return
        
        avg_start_key_point = self.__get_avg_keypoint(self.key_points, 0, Observer.COUNT_AVG_FRAME)
        frame_count = len(self.key_points)
        avg_end_key_point = self.__get_avg_keypoint(self.key_points, frame_count - Observer.COUNT_AVG_FRAME, frame_count - 1)

        avg_key_points = [avg_start_key_point, avg_end_key_point]
        self.cal_imag_engine.calculate_images(avg_key_points)

        return self.cal_imag_engine.get_distance()

if __name__ == "__main__":
    import time
    import numpy as np

    # Blender 카메라 내부 행렬 (K)
    camera_matrix = np.array([
        [2666.67, 0, 960],
        [0, 2666.67, 540],
        [0, 0, 1]
    ], dtype=np.float32)

    import os
    start = time.perf_counter()
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, "Phone_White_100mm_1.mp4")

    observer = Observer(6, camera_matrix)
    observer.parse_video(file_path)
    print(f"Distance : {observer.get_distance()}")
    print(f"Displacement : {observer.get_displacement()}")

    # print(observer.get_camera_names())

    # if observer.activate_camera("USB CAMERA"):
    #     print("Active Camera")
    
    #     if observer.start_observe():
    #         print("Start Observer")

    #         time.sleep(5)

    #         if observer.stop_observe():
    #             print("Stop Observer")
    #             print(f"Distance : {observer.get_distance()}")
    #         else:
    #             print("Err3")

    #     else:
    #         print("Err2")

    # else:
    #     print("Err1")
