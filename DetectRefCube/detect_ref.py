from ultralytics import YOLO
import cv2
import os
import numpy as np

class DetectModule:

    MODEL_PATH = "best.pt"

    def __init__(self, input_video_path:str):
        # 모델 경로 설정
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, DetectModule.MODEL_PATH)

        # 모델 로드
        self.model = YOLO(model_path)

        # 📹 영상 파일 경로 (웹캠 사용 시 0)
        self.cap = cv2.VideoCapture(input_video_path)  # 웹캠을 사용하려면 `video_path = 0`으로 변경

        # 관절 위치 저장을 위한 딕셔너리
        self.alpha = 0.2  # 지수 이동 평균 가중치 (0.1~0.3 추천)
        self.smoothed_keypoints = {}

        # 감지된 key의 좌표 저장 리스트
        """
            Example)
            [[[     1142.8      601.53     0.99995]  == (0,0,0)
              [     1158.9      342.49     0.99995]  == (0,0,n)
              [     997.94      783.08     0.99995]  == (0,n,0)
              [       1433      701.01     0.99995]]] == (n,0,0)
        """
        self.key_points = []

        self.__analyze_video()

    def __del__(self):
        self.cap.release()

    def __analyze_video(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # YOLO Pose 추론
            results = self.model(frame)

            for result in results:
                keypoints = result.keypoints  # 관절 좌표 추출
                if keypoints is not None:
                    keypoints = keypoints.data.cpu().numpy()  # (1, 4, 3) → (4, 3) 형태로 변환  # GPU 텐서를 CPU로 옮긴 후 NumPy 변환

                    if keypoints.size == 0:
                        break

                    # 각 관절 포인트 보정 (EMA 적용)
                    for i, (x, y, conf) in enumerate(keypoints[0]):  # 첫 번째 객체만 사용
                        if i not in self.smoothed_keypoints:
                            self.smoothed_keypoints[i] = (x, y)  # 처음에는 원본 좌표 사용
                        else:
                            prev_x, prev_y = self.smoothed_keypoints[i]
                            self.smoothed_keypoints[i] = (
                                self.alpha * x + (1 - self.alpha) * prev_x,
                                self.alpha * y + (1 - self.alpha) * prev_y,
                            )

                    # 보정된 관절 좌표를 다시 적용
                    keypoints[0] = np.array(
                        [[self.smoothed_keypoints[i][0], self.smoothed_keypoints[i][1], conf]
                        for i in range(len(keypoints[0]))]
                    )

                    self.key_points.append(keypoints[0])

    def get_key_points(self):
        return self.key_points
    
if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, "test_input_video.mp4")

    detectModule = DetectModule(file_path)

    print(detectModule.get_key_points())