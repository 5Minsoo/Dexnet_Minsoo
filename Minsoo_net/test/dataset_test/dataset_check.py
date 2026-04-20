import zarr
import numpy as np
import logging
import random
import matplotlib.pyplot as plt

zarr_path = '/home/minsoo/Dexnet_Minsoo/grasp_dataset_big1.zarr'
root = zarr.open(str(zarr_path), mode="r")

threshold = 0.002

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('zarr').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

print(f'파라미터: {root.attrs["config"]}')
def print_stats(arr):
    logging.debug(f'  유효 샘플 수(>0): {len(arr)}')
    if len(arr) > 0:
        logging.debug(f'  max:    {arr.max():.5f}')
        logging.debug(f'  75%:    {np.percentile(arr, 75):.5f}')
        logging.debug(f'  median: {np.median(arr):.5f}')
        logging.debug(f'  25%:    {np.percentile(arr, 25):.5f}')
        logging.debug(f'  min:    {arr.min():.5f}')
        logging.debug(f'  mean:   {arr.mean():.5f}')
        logging.debug(f'  std:    {arr.std():.5f}')

all_labels = []
dataset_indices = []

# 전체 통계 계산용 변수
global_total_samples = 0
global_success_samples = 0

for obj_key in root.keys():
    obj_group = root[obj_key]
    
    # 물체별 통계 계산용 변수
    obj_labels_list = []
    obj_total_samples = 0
    obj_success_samples = 0
    
    for pose_key in obj_group.keys():
        labels = np.array(obj_group[pose_key]["labels"])
        num_samples = len(labels)
        
        # 1. 물체별 데이터 개수 누적
        obj_total_samples += num_samples
        obj_success_samples += np.sum(labels > threshold)
        
        # 2. 통계 출력용 유효 데이터(0 초과) 모으기
        valid_labels = labels[labels > 0]
        if len(valid_labels) > 0:
            obj_labels_list.append(valid_labels)
            
        # 3. 뷰어용 인덱스 저장
        for i in range(num_samples):
            dataset_indices.append((obj_key, pose_key, i))

    # 전역(Global) 변수에 현재 물체의 통계 누적
    global_total_samples += obj_total_samples
    global_success_samples += obj_success_samples

    # 물체별 통계 및 정답 비율 출력
    logging.debug(f'\n[{obj_key}]')
    if obj_labels_list:
        obj_labels_concat = np.concatenate(obj_labels_list)
        all_labels.append(obj_labels_concat)
        print_stats(obj_labels_concat)
        
    if obj_total_samples > 0:
        obj_success_rate = (obj_success_samples / obj_total_samples) * 100
        logging.debug(f'  => {obj_key} 정답 비율: {obj_success_samples}/{obj_total_samples} ({obj_success_rate:.2f}%)')
    else:
        logging.debug(f'  => {obj_key} 정답 비율: 데이터 없음')

logging.debug('\n' + '=' * 40)
logging.debug('[전체 통계]')
if all_labels:
    print_stats(np.concatenate(all_labels))

# 총 정답 비율 출력
if global_total_samples > 0:
    global_success_rate = (global_success_samples / global_total_samples) * 100
    logging.debug(f'총 정답 비율: {global_success_samples}/{global_total_samples} ({global_success_rate:.2f}%)')
else:
    logging.debug('총 정답 비율: 데이터 없음')

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
        
    obj_key, pose_key, sample_idx = random.choice(dataset_indices)
    
    img = root[obj_key][pose_key]["images"][sample_idx]
    label = root[obj_key][pose_key]["labels"][sample_idx]
    
    img_display = np.squeeze(img)
    
    ax.clear()
    ax.imshow(img_display, cmap='gray')
    
    is_success = "SUCCESS" if label > threshold else "FAIL"
    ax.set_title(f"Obj: {obj_key} | Pose: {pose_key} | Idx: {sample_idx}\nLabel: {label:.5f} ({is_success})")
    ax.axis('off')
    
    plt.draw()
    
    wait = plt.waitforbuttonpress()
    if wait is None:
        print("뷰어 창이 닫혀 프로그램을 종료합니다.")
        break