import numpy as np
import cv2
# 파일 로드
data = np.load('/home/minsoo/Dexnet_Minsoo/Minsoo_net/test/saved_data/raw_depth_data_1.npz')

# 어떤 데이터(Key)들이 들어있는지 확인
print("들어있는 항목들:", data.files)

# # 특정 데이터 꺼내기 (아까 저장할 때 지정한 이름들)
depth = data['depth']
# cropped = data['cropped']
# poses = data['poses']
# samples = data['samples']
# success_probs = data['success_probs']

# 형태(shape) 확인
print(f"Depth shape: {depth.shape}")

cv2.imshow('depth',depth)
cv2.waitKey(0)
# 작업이 끝난 후 닫아주는 것이 좋습니다 (메모리 관리)
data.close()