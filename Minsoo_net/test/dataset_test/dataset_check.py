import zarr
import numpy as np
zarr_path='/home/minsoo/Dexnet_Minsoo/grasp_dataset_biggest.zarr'
root = zarr.open(str(zarr_path), mode="r")
paths = []
success = 0  
total_samples = 0 # 전체 샘플 수를 저장할 변수 추가

def print_stats(arr):
    print(f'  샘플 수: {len(arr)}')
    print(f'  max:    {arr.max():.5f}')
    print(f'  75%:    {np.percentile(arr, 75):.5f}')
    print(f'  median: {np.median(arr):.5f}')
    print(f'  25%:    {np.percentile(arr, 25):.5f}')
    print(f'  min:    {arr.min():.5f}')
    print(f'  mean:   {arr.mean():.5f}')
    print(f'  std:    {arr.std():.5f}')

all_labels = []

for obj_key in root.keys():
    obj_group = root[obj_key]
    obj_labels = []
    for pose_key in obj_group.keys():
        labels = np.array(obj_group[pose_key]["labels"])
        labels = labels[labels > 0]
        obj_labels.append(labels)
    obj_labels = np.concatenate(obj_labels)
    all_labels.append(obj_labels)
    print(f'[{obj_key}]')
    print_stats(obj_labels)

print('=' * 40)
print('[전체]')
print_stats(np.concatenate(all_labels))
