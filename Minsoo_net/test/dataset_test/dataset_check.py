import zarr
import numpy as np
import logging
import random
import matplotlib.pyplot as plt

zarr_path = '/home/minsoo/Dexnet_Minsoo/grasp_dataset.zarr'
root = zarr.open(str(zarr_path), mode="r")

success = 0  
total_samples = 0
threshold = 0.032
labels_positive = 0
label_len = 0

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('zarr').setLevel(logging.WARNING)

def print_stats(arr):
    logging.debug(f'  샘플 수: {len(arr)}')
    logging.debug(f'  max:    {arr.max():.5f}')
    logging.debug(f'  75%:    {np.percentile(arr, 75):.5f}')
    logging.debug(f'  median: {np.median(arr):.5f}')
    logging.debug(f'  25%:    {np.percentile(arr, 25):.5f}')
    logging.debug(f'  min:    {arr.min():.5f}')
    logging.debug(f'  mean:   {arr.mean():.5f}')
    logging.debug(f'  std:    {arr.std():.5f}')

all_labels = []

# 이미지 샘플링을 위한 전체 데이터 인덱스 저장 리스트
# (obj_key, pose_key, index) 형태로 저장
dataset_indices = []

for obj_key in root.keys():
    obj_group = root[obj_key]
    obj_labels = []
    
    for pose_key in obj_group.keys():
        labels = np.array(obj_group[pose_key]["labels"])
        num_samples = len(labels)
        label_len += num_samples
        
        # 유효한 label(0 초과) 통계용
        valid_labels = labels[labels > 0]
        obj_labels.append(valid_labels)
        labels_positive += np.sum(valid_labels > threshold)
        
        # 시각화를 위해 모든 샘플의 위치(인덱스)를 기록
        for i in range(num_samples):
            dataset_indices.append((obj_key, pose_key, i))

    if obj_labels:
        obj_labels_concat = np.concatenate(obj_labels)
        all_labels.append(obj_labels_concat)
        logging.debug(f'[{obj_key}]')
        print_stats(obj_labels_concat)

logging.debug('=' * 40)
logging.debug('[전체 통계]')
if all_labels:
    print_stats(np.concatenate(all_labels))
logging.debug(f'정답 비율: {labels_positive/label_len*100:.2f}%')

# ==========================================
# 랜덤 이미지 뷰어 기능 추가 부분
# ==========================================
print("\n[뷰어 시작] 사진 창을 띄웁니다.")
print("아무 키나 누르면 다음 랜덤 이미지가 표시됩니다. (창을 닫으면 종료)")

fig, ax = plt.subplots(figsize=(6, 6))

while True:
    if not dataset_indices:
        print("시각화할 데이터가 없습니다.")
        break
        
    # 인덱스 리스트에서 랜덤하게 하나 추출
    obj_key, pose_key, sample_idx = random.choice(dataset_indices)
    
    # Zarr에서 해당 이미지와 라벨만 로드 (Lazy Loading)
    img = root[obj_key][pose_key]["images"][sample_idx]
    label = root[obj_key][pose_key]["labels"][sample_idx]
    
    # 채널 차원이 (H, W, 1)일 경우 matplotlib을 위해 (H, W)로 스퀴즈
    img_display = np.squeeze(img)
    
    ax.clear()
    ax.imshow(img_display, cmap='gray') # Depth 이미지일 경우 흑백 컬러맵 사용
    
    # 성공 기준(threshold)을 넘었는지 판단하여 타이틀에 표시
    is_success = "SUCCESS" if label > threshold else "FAIL"
    ax.set_title(f"Obj: {obj_key} | Pose: {pose_key} | Idx: {sample_idx}\nLabel: {label:.5f} ({is_success})")
    ax.axis('off')
    
    plt.draw()
    
    # 키보드 입력을 기다림. 창을 닫으면(False 반환) 루프 탈출
    wait = plt.waitforbuttonpress()
    if wait is None:
        print("뷰어 창이 닫혀 프로그램을 종료합니다.")
        break