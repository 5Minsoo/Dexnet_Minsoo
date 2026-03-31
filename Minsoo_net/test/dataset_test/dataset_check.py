import zarr
import numpy as np
import logging

zarr_path='/home/minsoo/Dexnet_Minsoo/grasp_dataset.zarr'
root = zarr.open(str(zarr_path), mode="r")
paths = []
success = 0  
total_samples = 0 # 전체 샘플 수를 저장할 변수 추가
threshold=0.032
labels_positive=0
label_len=0
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

for obj_key in root.keys():
    obj_group = root[obj_key]
    obj_labels = []
    for pose_key in obj_group.keys():
        labels = np.array(obj_group[pose_key]["labels"])
        label_len+=len(labels)
        labels = labels[labels > 0]
        obj_labels.append(labels)
        labels_positive+=np.sum(labels>threshold)

    obj_labels = np.concatenate(obj_labels)
    all_labels.append(obj_labels)
    logging.debug(f'[{obj_key}]')
    print_stats(obj_labels)

logging.debug('=' * 40)
logging.debug('[전체]')
print_stats(np.concatenate(all_labels))
logging.debug(f'정답 비율: {labels_positive/label_len*100:.2f}%')
