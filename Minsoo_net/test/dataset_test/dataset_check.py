import zarr
import numpy as np
import logging
import random
import matplotlib.pyplot as plt
import trimesh
import multiprocessing as mp
from pathlib import Path
from matplotlib.colors import Normalize

zarr_path = '/home/minsoo/Dexnet_Minsoo/grasp_dataset_ABC.zarr'
mesh_root = Path('/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/object/Frankapanda')
MESH_EXTS = ('.obj', '.stl', '.ply', '.OBJ', '.STL', '.PLY')

root = zarr.open(str(zarr_path), mode="r")

threshold = 0.002

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('zarr').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('trimesh').setLevel(logging.WARNING)

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

# ==========================================
# 메쉬 인덱스 (시작할 때 한 번만 스캔)
# ==========================================
def build_mesh_index(root_dir, exts=MESH_EXTS):
    """root_dir 하위를 재귀적으로 훑어서 {파일이름(확장자제외): 절대경로} dict 생성."""
    index = {}
    for ext in exts:
        for p in Path(root_dir).rglob(f'*{ext}'):
            index[p.stem] = p  # 파일명(확장자 제외)이 obj_key와 매칭됨
    return index

print(f'\n[메쉬 인덱싱] {mesh_root} 스캔 중...')
mesh_index = build_mesh_index(mesh_root)
print(f'[메쉬 인덱싱] 총 {len(mesh_index)}개 메쉬 파일 발견')

# ==========================================
# 메쉬 뷰어 (별도 프로세스에서 띄움 → 메인 루프 블록 안 됨)
# ==========================================
def _show_mesh_proc(mesh_path, title):
    """별도 프로세스에서 trimesh 뷰어 실행."""
    mesh = trimesh.load(str(mesh_path), force='mesh')
    scene = trimesh.Scene(mesh)
    scene.show(caption=title)  # 이 프로세스에서 블록되지만 메인엔 영향 X

_current_mesh_proc = None

def show_mesh(obj_key):
    """obj_key에 해당하는 메쉬를 별도 창으로 띄움. 이전 창은 자동 종료."""
    global _current_mesh_proc

    # 이전 메쉬 창 종료
    if _current_mesh_proc is not None and _current_mesh_proc.is_alive():
        _current_mesh_proc.terminate()
        _current_mesh_proc.join(timeout=1)

    mesh_path = mesh_index.get(obj_key)
    if mesh_path is None:
        print(f"  [경고] '{obj_key}'에 해당하는 메쉬를 찾을 수 없습니다.")
        return

    _current_mesh_proc = mp.Process(
        target=_show_mesh_proc,
        args=(mesh_path, f'{obj_key} ({mesh_path.suffix})'),
        daemon=True,
    )
    _current_mesh_proc.start()

# ==========================================
# 통계 부분 (기존 코드 그대로)
# ==========================================
all_labels = []
dataset_indices = []

global_total_samples = 0
global_success_samples = 0
obj_num = 0

for obj_key in root.keys():
    obj_group = root[obj_key]

    obj_labels_list = []
    obj_total_samples = 0
    obj_success_samples = 0
    obj_num += 1
    for pose_key in obj_group.keys():
        labels = np.array(obj_group[pose_key]["labels"])
        images=np.array(obj_group[pose_key]['images'])
        num_samples = len(labels)

        obj_total_samples += num_samples
        obj_success_samples += np.sum(labels > threshold)

        valid_labels = labels[labels > 0]
        if len(valid_labels) > 0:
            obj_labels_list.append(valid_labels)

        for i in range(num_samples):
            dataset_indices.append((obj_key, pose_key, i))

    global_total_samples += obj_total_samples
    global_success_samples += obj_success_samples

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
logging.debug(f'물체 개수: {obj_num}')
if all_labels:
    print_stats(np.concatenate(all_labels))

if global_total_samples > 0:
    global_success_rate = (global_success_samples / global_total_samples) * 100
    logging.debug(f'총 정답 비율: {global_success_samples}/{global_total_samples} ({global_success_rate:.2f}%)')
else:
    logging.debug('총 정답 비율: 데이터 없음')

# ==========================================
# 랜덤 이미지 + 메쉬 뷰어
# ==========================================
print("\n[뷰어 시작] 사진 창 + 메쉬 창이 함께 뜹니다.")
print("matplotlib 창에서 아무 키나 누르면 다음 랜덤 샘플로 넘어갑니다. (창을 닫으면 종료)")

if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(6, 6))

    last_obj_key = None  # 같은 obj면 메쉬 창 재사용
    try:
        while True:
            if not dataset_indices:
                print("시각화할 데이터가 없습니다.")
                break

            obj_key, pose_key, sample_idx = random.choice(dataset_indices)

            img = root[obj_key][pose_key]["images"][sample_idx]
            label = root[obj_key][pose_key]["labels"][sample_idx]

            img_display = np.squeeze(img).astype(np.float32)

            # ── 대비 향상: 1~99 퍼센타일로 clip해서 outlier 제거 ──
            valid = img_display[img_display > 0] if (img_display > 0).any() else img_display
            vmin, vmax = np.percentile(valid, [1, 99])
            if vmax - vmin < 1e-6:        # 거의 평탄한 이미지 보호
                vmin, vmax = img_display.min(), img_display.max() + 1e-6

            ax.clear()
            ax.imshow(
                img_display,
                cmap='turbo',             # 'viridis', 'plasma', 'magma'도 좋음
                vmin=vmin, vmax=vmax,
            )

            is_success = "SUCCESS" if label > threshold else "FAIL"
            ax.set_title(f"Obj: {obj_key} | Pose: {pose_key} | Idx: {sample_idx}\n"
                        f"Label: {label:.5f} ({is_success}) | depth: [{vmin:.3f}, {vmax:.3f}]")
            ax.axis('off')
            plt.draw()

            if obj_key != last_obj_key:
                show_mesh(obj_key)
                last_obj_key = obj_key

            wait = plt.waitforbuttonpress()
            if wait is None:
                print("뷰어 창이 닫혀 프로그램을 종료합니다.")
                break
    finally:
        if _current_mesh_proc is not None and _current_mesh_proc.is_alive():
            _current_mesh_proc.terminate()
            _current_mesh_proc.join(timeout=1)