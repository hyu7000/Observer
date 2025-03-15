def calculate_camera_depth(actual_distance_cm, pixel_distance_px, focal_length_px):
    """
    카메라 내부 파라미터를 알고 있을 때, 두 점의 실제 거리와 픽셀 거리를 이용하여
    카메라와 점들 사이의 깊이(Z)를 계산하는 함수.
    
    Parameters:
    - actual_distance_cm: 두 점의 실제 거리 (cm 단위)
    - pixel_distance_px: 이미지에서 두 점 사이의 픽셀 거리 (px 단위)
    - focal_length_px: 카메라의 초점 거리 (픽셀 단위, fx 값)
    
    Returns:
    - 카메라의 깊이 Z (cm)
    """
    Z = (actual_distance_cm * focal_length_px) / pixel_distance_px
    return Z

# 예제 데이터
actual_distance_cm = 6  # 두 점 사이의 실제 거리 (6cm)
pixel_distance_px = 1280-960  # 두 점의 픽셀 거리 
focal_length_px = 2666.67   # 카메라 초점 거리 (fx = 1500px, 예제)

# 카메라 깊이 Z 계산
camera_depth = calculate_camera_depth(actual_distance_cm, pixel_distance_px, focal_length_px)
print(f"📌 카메라와 수직거리 (깊이, Z): {camera_depth:.2f} cm")
