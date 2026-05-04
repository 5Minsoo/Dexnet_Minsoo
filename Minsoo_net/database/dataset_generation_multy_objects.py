import sys
from pathlib import Path
import time
import argparse

import numpy as np
import zarr
import cv2
import yaml
import trimesh

from Minsoo_net.grasp.rendering import GraspRenderer
from Minsoo_net.grasp import GraspPipeline

parser = argparse.ArgumentParser(description="데이터셋 생성")
parser.add_argument("--mode", "-m", default="continue", help="restart or continue")
args = parser.parse_args()

# --- 설정 ---
use_visual = True

with open('/home/minsoo/Dexnet_Minsoo/Minsoo_net/config/master_config.yaml') as f:
    config = yaml.safe_load(f)
    num_grasps = config.get("num_grasps", 300)
    quality_threshold = config.get("quality_threshold", 0.002)
    prob_threshold = config.get("stable_pose_prob_threshold", 0.012)
    num_stable_poses = config.get("num_stable_poses", 10)
    max_angle = config.get("max_angle_deg", 15)
    zarr_path = config.get("zarr_path", "grasp_dataset.zarr")
    co = config['cam_offset']
    camera_offsets = np.linspace(co['start'], co['stop'], co['num'])

zarr_path = '/home/minsoo/Dexnet_Minsoo/grasp_dataset_test.zarr'
num_grasps = 10
output_size = 32
crop_size = 96
batch_size = 2048
batch_flush = 512

# --- 단일 mesh 파일 지정 ---
mesh_path = Path('/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/object/Frankapanda/3/00576648_d4d00869b27c21532c4f2e7b_step_000_0000.obj')
object_name = mesh_path.stem
mesh_path_str = str(mesh_path)

if args.mode == "restart":
    print(f"기존 데이터를 삭제하고 새로 작업을 시작합니다. {zarr_path}")
    store = zarr.open(zarr_path, mode='w')
elif args.mode == "continue":
    print(f"기존 데이터에 이어서 작업을 시작합니다. {zarr_path}")
    store = zarr.open(zarr_path, mode='a')
    if object_name in store and store[object_name].attrs.get("done", False):
        print(f"이미 완료된 물체입니다: {object_name}")
        sys.exit()
else:
    sys.exit()

store.attrs["config"] = config

tmp_imgs, tmp_labels, tmp_z = [], [], []

def flush_to_zarr(img_ds, label_ds, z_ds):
    """버퍼의 데이터를 zarr에 한 번에 추가"""
    global tmp_imgs, tmp_labels, tmp_z
    if not tmp_imgs:
        return
    img_ds.append(np.array(tmp_imgs))
    label_ds.append(np.array(tmp_labels))
    z_ds.append(np.array(tmp_z))
    tmp_imgs, tmp_labels, tmp_z = [], [], []

def get_camera_positions(grasp, pose, offsets=[0.2, 0.25, 0.3]):
    T_world_gripper = pose @ grasp.T_grasp_obj
    z_axis = T_world_gripper[:3, 2]
    gripper_pos = T_world_gripper[:3, 3]
    cam_target = gripper_pos
    cam_positions = [gripper_pos - z_axis * d for d in offsets]
    return cam_positions, cam_target

# --- 단일 물체 처리 ---
m = trimesh.load(mesh_path_str, force='mesh')
if min(m.bounding_box_oriented.extents) < 0.005:
    print('너무 얇아서 제외')
    sys.exit()

print(f'{object_name} 로드중')

grasp_pipeline = GraspPipeline(
    mesh_path_str,
    quality_threshold=quality_threshold,
    num_grasps=num_grasps,
    max_approach_angle_deg=max_angle,
    num_poses=num_stable_poses,
)
renderer = GraspRenderer(mesh_path_str)
if use_visual:
    viewer = renderer.scene.create_viewer()
    for _ in range(10):
        viewer.render()

obj_group = store.require_group(object_name)

existing_pose_keys = sorted(
    [k for k in obj_group.keys() if k.startswith("pose")],
    key=lambda x: int(x.replace("pose", ""))
)
start_idx = 0
for pose_key in existing_pose_keys:
    pose_num = int(pose_key.replace("pose", ""))
    labels_len = obj_group[pose_key]['labels'].shape[0]
    if labels_len < 3:
        start_idx = pose_num
        print(f"  [!] {pose_key} 데이터 부족 ({labels_len}개). 여기서부터 재시작합니다.")
        break
    else:
        start_idx = pose_num + 1

finish = time.time()
# --- 메인 루프 ---
for pose, failed_grasps, quality_grasps, quality_scores in grasp_pipeline.execute(start_index=start_idx):
    start = time.time()
    print(f'Grasp sampling 시간 {int(start-finish)}초')
    renderer.set_stable_pose(pose)
    pose_group = obj_group.require_group(f"pose{start_idx}")
    start_idx += 1
    img_ds = pose_group.create_array("images", shape=(0, output_size, output_size),
                                     chunks=(batch_size, output_size, output_size),
                                     dtype='float32', overwrite=True)
    label_ds = pose_group.create_array("labels", shape=(0,),
                                       chunks=(batch_size,),
                                       dtype='float32', overwrite=True)
    z_ds = pose_group.create_array("gripper_depth", shape=(0,),
                                   chunks=(batch_size,),
                                   dtype='float32', overwrite=True)
    # 성공(quality) / 실패(0) 데이터 분류
    tasks = [(quality_grasps, quality_scores),
             (failed_grasps, [0.0]*len(failed_grasps))]

    for grasps, labels in tasks:
        print("이미지 랜더링 시작 (진행중..)")
        for grasp, label in zip(grasps, labels):
            cam_poses, cam_target = get_camera_positions(grasp, pose, offsets=camera_offsets)
            metalic, roughness = renderer.sample_material()
            renderer.set_material(metalic=metalic, roughness=roughness)
            for cam_pos in cam_poses:
                depth = renderer.render(camera_pos=cam_pos, target_pos=cam_target)
                origin = [0, 0, 0]
                center = (pose @ np.append(grasp.center, 1.0))[:3]
                axis = (pose @ np.append(grasp.axis, 1.0))[:3]
                image_point = renderer.world_to_pixel([origin, center, axis])
                grasp_depth = (renderer.get_extrinsic() @ np.append(center, 1.0))[2]

                cropped = GraspRenderer.crop_grasp_image(
                    depth, image_point[1], image_point[2]-image_point[0],
                    crop_size=crop_size, output_size=output_size
                )
                if use_visual:
                    renderer.scene.step()
                    renderer.scene.update_render()
                    renderer.sensor.take_picture()
                    viewer.render()

                    center_2d = image_point[1]
                    axis_2d = image_point[2] - image_point[0]

                    # ─── 흑백 유지하면서 3채널로 ───
                    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_bgr = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)

                    cropped_norm = cv2.normalize(cropped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    cropped_bgr = cv2.cvtColor(cropped_norm, cv2.COLOR_GRAY2BGR)

                    depth_debug = renderer.draw_grasp_debug(
                        image=depth_bgr, center=center_2d, axis=axis_2d, depth=grasp_depth
                    )
                    cropped_debug = renderer.draw_cropped_debug(
                        cropped=cropped_bgr,
                        depth=grasp_depth,
                        crop_size=crop_size,
                        output_size=output_size,
                    )

                    h = depth_debug.shape[0]
                    cropped_resized = cv2.resize(cropped_debug, (h, h), interpolation=cv2.INTER_NEAREST)
                    combined_img = cv2.hconcat([depth_debug, cropped_resized])

                    cv2.imshow('Depth vs Cropped', depth_debug)
                    cv2.imshow('Cropped',cropped_debug)
                    while True:
                        renderer.scene.step()
                        renderer.scene.update_render()
                        viewer.render()
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key != 255:  # 아무 키나 눌리면 (255는 키 안 눌렸을 때)
                            break

                print('image cropped')
                tmp_imgs.append(cropped)
                tmp_labels.append(label)
                tmp_z.append(grasp_depth)
                if len(tmp_imgs) >= batch_flush:
                    flush_to_zarr(img_ds, label_ds, z_ds)
                    print(f'flushed_{object_name}_pose{start_idx}')
    flush_to_zarr(img_ds, label_ds, z_ds)
    finish = time.time()
    print(f'이미지 랜더링 종료 Pose{start_idx-1} 걸린시간: {int(finish-start)}초')

obj_group.attrs["done"] = True
print(f'✓ {object_name} 완료')

print(f"Zarr 데이터셋 생성 완료! 경로: {zarr_path}")