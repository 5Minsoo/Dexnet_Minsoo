import numpy as np
import matplotlib.pyplot as plt

# 1. 파라미터 설정
r = np.linspace(0,5,10)  # 반지름 (고정값 혹은 범위)
theta = np.deg2rad(np.linspace(0, 360, 20))  # 0~360도 (한바퀴)
phi = np.deg2rad(np.linspace(0, 30, 10))     # 0~90도 (위에서 옆면까지)

# 2. 그리드 생성 (각도의 조합을 만듦)
r,THETA, PHI = np.meshgrid(r,theta, phi)

# 3. 구면 좌표 -> 직교 좌표 변환 (수식 적용)
# x = r * sin(phi) * cos(theta)
# y = r * sin(phi) * sin(theta)
# z = r * cos(phi)
X = r * np.sin(PHI) * np.cos(THETA)
Y = r * np.sin(PHI) * np.sin(THETA)
Z = r * np.cos(PHI)

# 4. 시각화
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 점(scatter)으로 그리기
ax.scatter(X, Y, Z, c=Z, cmap='magma', s=20)

# 축 레이블 설정
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('Viewpoint Sampling (Dome Shape)')

plt.show() # fig.show() 대신 plt.show()를 권장합니다.