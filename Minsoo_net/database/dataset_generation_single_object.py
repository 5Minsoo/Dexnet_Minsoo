from Minsoo_net.grasp.rendering import GraspRenderer
from Minsoo_net.grasp import GraspPipeline
import numpy as np
import zarr
import cv2
from pathlib import Path
# --- 설정 ---
use_visual=True
zarr_path = "grasp_dataset.zarr"
output_size = 64
batch_size = 100 

num_grasps=100
quality_threshold=0.03
num_camera_points=50
camera_radius=(0.3,0.6)

print("기존 데이터에 이어서 작업을 시작합니다.")
store = zarr.open(zarr_path, mode='a')

mesh_path=Path(__file__).parent.parent.resolve()

tmp_imgs, tmp_labels,tmp_z = [], [],[]

def flush_to_zarr(img_ds,label_ds,z_ds):
    """버퍼의 데이터를 zarr에 한 번에 추가"""
    global tmp_imgs, tmp_labels, tmp_z
    if not tmp_imgs: return
    
    # zarr의 .append()는 축 0을 기준으로 자동으로 확장하며 붙여줍니다.
    img_ds.append(np.array(tmp_imgs))
    label_ds.append(np.array(tmp_labels))
    z_ds.append(np.array(tmp_z))
    
    tmp_imgs, tmp_labels,tmp_z = [], [],[]


object_name=mesh_path.stem
mesh_path=str(mesh_path)
print(f'{object_name} 로드중')

grasp_pipeline=GraspPipeline(mesh_path,quality_threshold=quality_threshold,num_grasps=num_grasps)
renderer=GraspRenderer(mesh_path)
points=GraspRenderer.sample_spherical_positions(camera_radius,(0,3.14/6),(0,3.14),num_points=num_camera_points)

obj_group=store.require_group(object_name)

existing_pose_keys = sorted(
[k for k in obj_group.keys() if k.startswith("pose")],
key=lambda x: int(x.replace("pose", ""))
)
start_idx=0

# --- 메인 루프 ---
for pose, collision_free_grasps, quality_grasps, quality_scores in grasp_pipeline.execute(start_index=start_idx):
    renderer.set_stable_pose(pose)
    pose_group=obj_group.require_group(f"pose{start_idx}")
    start_idx+=1
    img_ds = pose_group.create_array("images", shape=(0, output_size, output_size), 
                            chunks=(batch_size, output_size, output_size), 
                            dtype='uint8',overwrite=True)
    label_ds = pose_group.create_array("labels", shape=(0,), 
                                    chunks=(batch_size,), 
                                    dtype='float32',overwrite=True)
    z_ds = pose_group.create_array("gripper_depth", shape=(0,), 
                                    chunks=(batch_size,), 
                                    dtype='float32',overwrite=True)
    # 성공(quality) / 실패(0) 데이터 분류
    tasks = [(quality_grasps, quality_scores), 
            (collision_free_grasps, [0.0]*len(collision_free_grasps))]

    for grasps, labels in tasks:
        for grasp, label in zip(grasps, labels):
            for point in points:
                depth = renderer.render(camera_pos=point)
                
                # 좌표 계산 및 시각화 (기존 로직)
                origin = [0, 0, 0]
                center = (pose @ np.append(grasp.center, 1.0))[:3]
                axis = (pose @ np.append(grasp.axis, 1.0))[:3]
                image_point = renderer.world_to_pixel([origin, center, axis])
                
                cropped = GraspRenderer.crop_grasp_image(
                    depth, image_point[1], image_point[2]-image_point[0], 
                    crop_size=96, output_size=output_size
                )
                if use_visual:
                    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                    
                    # 원본 이미지 위에 Grasp 중심점과 축 그리기 (디버그용)
                    center_2d = image_point[1]
                    axis_2d = image_point[2] - image_point[0]
                    depth_debug = GraspRenderer.draw_grasp_debug(depth_color, center_2d, axis_2d)

                    # 2. 크롭 뎁스 정규화 및 컬러맵 적용
                    cropped_norm = cv2.normalize(cropped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    cropped_color = cv2.applyColorMap(cropped_norm, cv2.COLORMAP_JET)

                    # 3. 크기 맞추기 및 이어붙이기 (hconcat)
                    # cv2.hconcat을 쓰려면 두 이미지의 세로 길이(h)가 같아야 하므로 크롭 이미지를 확대합니다.
                    h = depth_debug.shape[0]
                    cropped_resized = cv2.resize(cropped_color, (h, h), interpolation=cv2.INTER_NEAREST)
                    
                    # 가로로 이어 붙이기
                    combined_img = cv2.hconcat([depth_debug, cropped_resized])

                    # 4. 화면 출력
                    cv2.imshow('Depth vs Cropped', combined_img)
                    
                    # 디버깅 시 하나씩 확인하려면 waitKey(0), 자동으로 휙휙 넘어가게 하려면 waitKey(1)
                    cv2.waitKey(1)
                
                tmp_imgs.append(cropped)
                tmp_labels.append(label)
                tmp_z.append(center[2])
                if len(tmp_imgs) >= batch_size:
                    flush_to_zarr(img_ds,label_ds,z_ds)
    flush_to_zarr(img_ds,label_ds,z_ds)


print(f"Zarr 데이터셋 생성 완료! 경로: {zarr_path}")