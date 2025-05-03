from ultralytics import YOLO
import cv2
import os
import numpy as np

class DetectModule:

    MODEL_PATH = "best.pt"

    IS_DEBUG_MODE = True

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
              [     1158.9      342.49     0.99995]  == (n,0,0)
              [     997.94      783.08     0.99995]  == (0,n,0)
              [       1433      701.01     0.99995]]] == (0,0,n)
        """
        self.key_points = []

        # for debug
        if DetectModule.IS_DEBUG_MODE:
            self.output_dir = self.__init_debug()

        self.__analyze_video()

    def __init_debug(self):
        output_dir = "output_images"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  # 폴더 생성
        return output_dir  # 이미지 저장 폴더 경로 반환

    def __del__(self):
        self.cap.release()

    def __analyze_video(self):
        frame_count = 0  # 이미지 파일 이름을 위한 카운터
        
        while self.cap.isOpened():
            img = None            
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
                
                # 보정된 결과를 이미지에 반영                
                img = result.plot()  # 관절이 표시된 이미지 생성

             # 이미지 저장
            if img is not None:
                img_filename = os.path.join(self.output_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(img_filename, img)  # 이미지 파일로 저장
                frame_count += 1  # 프레임 번호 증가

                # 화면 출력 (선택 사항)
                cv2.imshow("Pose Detection", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키 누르면 종료
                    break

        self.__plot_keypoint_graph()

    def __plot_keypoint_graph(self):
        import matplotlib.pyplot as plt

        if not self.key_points:
            print("📭 시각화할 keypoint 데이터가 없습니다.")
            return

        kp_index = 2  # 확인하고 싶은 keypoint 인덱스

        xs = [p[kp_index][0] for p in self.key_points]
        ys = [p[kp_index][1] for p in self.key_points]
        confs = [p[kp_index][2] for p in self.key_points]

        frames = list(range(len(self.key_points)))

        fig, ax1 = plt.subplots()

        # 좌표 축
        ax1.set_xlabel("프레임 번호")
        ax1.set_ylabel("좌표 (px)", color="tab:blue")
        ax1.plot(frames, xs, label="X 좌표", color="tab:blue")
        ax1.plot(frames, ys, label="Y 좌표", color="tab:cyan", linestyle='--')
        ax1.tick_params(axis='y', labelcolor="tab:blue")

        # confidence 축 (우측)
        ax2 = ax1.twinx()
        ax2.set_ylabel("신뢰도 (confidence)", color="tab:red")
        ax2.plot(frames, confs, label="Confidence", color="tab:red", linestyle=':')
        ax2.tick_params(axis='y', labelcolor="tab:red")

        plt.title(f"Keypoint {kp_index} 좌표 및 신뢰도 변화")
        fig.legend(loc="upper left")
        fig.tight_layout()
        plt.grid(True)
        plt.show()

    def get_key_points(self):
        return self.key_points
    
if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    # file_path = os.path.join(base_path, "WIN_20250322_17_21_28_Pro.mp4")
    # file_path = os.path.join(base_path, "PhoneTestVideo.mp4")
    file_path = os.path.join(base_path, "PhoneTestVideo_Black.mp4")

    detectModule = DetectModule(file_path)

    key_points = detectModule.get_key_points()

    print(key_points)
    print(f'len : {len(key_points)}')