from CaptureRefCube.capture_ref import RecordModule
from CalculateImageEngine.calculate_engine import CalculateImageEngine
from DetectRefCube.detect_ref import DetectModule

class Observer:

    COUNT = 0

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

    def get_distance(self):
        self.cal_imag_engine.get_distance()


if __name__ == "__main__":
    import time
    import numpy as np

    # Blender 카메라 내부 행렬 (K)
    camera_matrix = np.array([
        [2666.67, 0, 960],
        [0, 2666.67, 540],
        [0, 0, 1]
    ], dtype=np.float32)

    start = time.perf_counter()

    observer = Observer(6, camera_matrix)

    print(observer.get_camera_names())

    if observer.activate_camera("USB CAMERA"):
        print("Active Camera")
    
        if observer.start_observe():
            print("Start Observer")

            time.sleep(1)

            if observer.stop_observe():
                print("Stop Observer")
            else:
                print("Err3")

        else:
            print("Err2")

    else:
        print("Err1")
