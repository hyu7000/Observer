import cv2
import threading
import win32com.client
import time
import os

class RecordModule:

    MAX_CHECK_CAMERA = 10
    MAX_RECORD_TIME = 300

    def __init__(self):
        self.cap = None

        self.width = None
        self.height = None
        self.fps = None

        self.is_recording = False  # 녹화 중 여부 플래그
        self.record_thread = None  # 녹화 스레드

    def get_camera_names(self):
        """Windows에서 DirectShow API를 사용하여 카메라 장치 이름 가져오기"""
        self.camera_names = []
        try:
            wmi = win32com.client.GetObject("winmgmts:\\\\.\\root\\cimv2")
            devices = wmi.ExecQuery("SELECT * FROM Win32_PnPEntity WHERE Name LIKE '%Camera%' OR Name LIKE '%Video%'")

            for device in devices:
                self.camera_names.append(device.Name)
        except Exception as e:
            print(f"에러 발생: {e}")
        
        return self.camera_names
    
    def activate_camera(self, camera_name) -> bool:
        self.get_camera_names()

        print(f'Camera list : {self.camera_names}')

        if len(self.camera_names) == 0:
            print("감지된 카메라 없음")
            return False
        
        if camera_name not in self.camera_names:
            print("정확하지 않은 카메라 이름")
            return False
        
        camera_index = self.camera_names.index(camera_name)

        self.cap = cv2.VideoCapture(0)

        # 웹캠 해상도 설정 (기본 해상도: 640x480)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30  # FPS 값이 0일 경우 기본값 30 사용

        if self.cap is not None:
            return True
        else:
            print("❌ 카메라 초기화 실패")
            return False

    def start_record(self, output_path:str) -> bool:
        if self.cap is None:
            return False
        
        if not self.cap.isOpened():
            print("❌ 카메라를 열 수 없습니다.")
            return False
        
        if self.is_recording:
            print("⚠️ 이미 녹화 중입니다.")
            return False
        
        self.is_recording = True

        self.record_thread = threading.Thread(target=self.__record, args=(output_path,))
        self.record_thread.start()
        return True

    def __record(self, output_path: str):
        """실제 녹화를 실행하는 내부 함수 (쓰레드에서 실행)"""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 코덱 설정
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        print("🎥 녹화를 시작합니다. 'stop_record()'를 호출하여 중지하세요.")

        self.abs_output_path = os.path.abspath(output_path)  # 절대 경로 변환

        print(f'abs_output_path : {self.abs_output_path}')

        start_time = time.time()

        while self.is_recording and self.cap.isOpened():
            elapsed_time = time.time() - start_time  # 경과 시간 계산

            if elapsed_time >= RecordModule.MAX_RECORD_TIME:  # 5분(300초) 초과 시 종료
                print("⏳ 5분이 경과하여 녹화를 자동 종료합니다.")
                break

            ret, frame = self.cap.read()
            if not ret:
                print("❌ 프레임을 가져올 수 없습니다. 녹화를 종료합니다.")
                break

            out.write(frame)

        out.release()
        print("🛑 녹화가 종료되었습니다.")

    def stop_record(self) -> bool:
        """녹화를 중지하는 함수"""
        if not self.is_recording:
            print("⚠️ 녹화 중이 아닙니다.")
            return False

        print("🛑 녹화를 중지하는 중...")
        self.is_recording = False
        self.record_thread.join()  # 쓰레드 종료 대기
        return True

    def __del__(self):
        """모든 자원 해제"""
        self.stop_record()

if __name__ == "__main__":
    recorder = RecordModule()

    # 카메라 이름 가져오기
    camera_names = recorder.get_camera_names()
    print(f"사용 가능한 카메라 목록: {camera_names}")

    # 카메라 활성화
    camera_names = recorder.activate_camera("USB CAMERA")

    if(recorder.start_record("output_video.mp4")):

        time.sleep(10)
        recorder.stop_record()
    else:
        print('not started')