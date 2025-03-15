import numpy as np

# 두 개의 벡터 정의
vector1 = np.array([-20.2248, 30.3010, 26.7109])
vector2 = np.array([-14.2248, 30.3010, 26.7109])

# 유클리드 거리 계산
distance = np.linalg.norm(vector2 - vector1)

# 결과 출력
print(f"두 벡터 간 이동 거리: {distance:.4f}")
