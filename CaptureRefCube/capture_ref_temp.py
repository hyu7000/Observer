import cv2

# 웹캠 열기 (기본 웹캠: 0, 여러 개 있을 경우 1, 2 등 조정 가능)
cap = cv2.VideoCapture(1)

# 웹캠 해상도 설정 (기본 해상도: 640x480)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # FPS 값이 0일 경우 기본값 30 사용

# 비디오 저장 설정
output_filename = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 코덱 설정 (MP4 형식)
out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

print("🎥 녹화를 시작합니다. 'q'를 눌러 종료하세요.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 가져올 수 없습니다. 녹화를 종료합니다.")
        break

    # 비디오 저장
    out.write(frame)

    # 화면에 출력 (실시간 미리보기)
    cv2.imshow("Webcam Recording", frame)

    # 'q'를 누르면 녹화 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("🛑 녹화를 종료합니다.")
        break

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
