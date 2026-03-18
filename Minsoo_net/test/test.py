from Minsoo_net.grasp.rendering import GraspRenderer
from Minsoo_net.grasp import GraspPipeline
import numpy as np
import zarr
import cv2
from pathlib import Path
# --- 설정 ---
zarr_path = "grasp_dataset.zarr"
output_size = 64
batch_size = 100 
mode=1 ## 1 for clean start  2 for resume

if mode ==1:
    print("기존 데이터를 초기화하고 처음부터 다시 시작합니다.")
    store = zarr.open(zarr_path, mode='w')
elif mode==2:
    print("기존 데이터에 이어서 작업을 시작합니다.")
    store = zarr.open(zarr_path, mode='a')

mesh_path=Path(__file__).parent.parent.resolve()
mesh_path=mesh_path/"data"/"object"
mesh_files = list(mesh_path.glob("*.obj")) + list(mesh_path.glob("*.stl"))

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


points=GraspRenderer.sample_spherical_positions((0.5,0.6),(0,3.14/6),(0,3.14),num_points=30)
for mesh_path in mesh_files:
    object_name=mesh_path.stem
    mesh_path=str(mesh_path)
    print(f'{object_name} 로드중')

    grasp_pipeline=GraspPipeline(mesh_path,quality_threshold=0.0,num_grasps=3)
    renderer=GraspRenderer(mesh_path)
    # renderer.sensor_config.uniqueness_ratio=45
    # print("uniqueness:", renderer.sensor_config.uniqueness_ratio)
    # print("block size:" ,renderer.sensor_config.block_height)
    # print("p1 penalty:", renderer.sensor_config.p1_penalty)
    # print("p2 penalty:", renderer.sensor_config.p2_penalty)

    obj_group=store.require_group(object_name)

    existing_pose_keys = sorted(
    [k for k in obj_group.keys() if k.startswith("pose")],
    key=lambda x: int(x.replace("pose", ""))
    )
    start_idx=0
    for pose_key in existing_pose_keys:
        pose_num = int(pose_key.replace("pose", ""))
        labels_len = obj_group[pose_key]['labels'].shape[0]
        
        # 데이터가 비어있거나 부족하면 해당 번호부터 다시 시작
        # Labels 에 저장된 라벨은 Points 와 관련돼있어서 훨씬더 길게 저장됨. num grasps가 경우에 따라 항상 더 커질수 있어서 일단 3개 미만으로 잡고 감. (3개 미만이면 그 pose부터 시작하게)
        if labels_len < 3: 
            start_idx = pose_num
            print(f"  [!] {pose_key} 데이터 부족 ({labels_len}개). 여기서부터 재시작합니다.")
            break
        else:
            # 정상이라면 다음 번호로 start_idx 갱신
            start_idx = pose_num +1

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
                    print('point: ',point)
                    depth = renderer.render(camera_pos=point)
                    
                    # 좌표 계산 (기존 로직)
                    origin = [0, 0, 0]
                    center = (pose @ np.append(grasp.center, 1.0))[:3]
                    axis = (pose @ np.append(grasp.axis, 1.0))[:3]
                    image_point = renderer.world_to_pixel([origin, center, axis])
                    
                    # 2D 상의 중심점과 방향 벡터 계산
                    center_2d = image_point[1]
                    axis_2d = image_point[2] - image_point[0]
                    
                    # ---------------------------------------------------------
                    # 1. 원본 Depth 이미지 처리 & 디버그 드로잉
                    # ---------------------------------------------------------
                    # 0~255 정규화 및 컬러맵 적용
                    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                    
                    # 컬러맵이 입혀진 이미지 위에 Grasp 중심과 축 그리기
                    depth_debug = GraspRenderer.draw_grasp_debug(depth_color, center_2d, axis_2d)
                    
                    # ---------------------------------------------------------
                    # 2. 크롭 이미지 처리
                    # ---------------------------------------------------------
                    cropped = GraspRenderer.crop_grasp_image(
                        depth, center_2d, axis_2d, 
                        crop_size=96, output_size=output_size
                    )

                    # ---------------------------------------------------------
                    # 3. 두 이미지 나란히 이어 붙이기 (hconcat)
                    # ---------------------------------------------------------
                    h, w = depth_debug.shape[:2]
                    # 크롭 이미지를 원본 이미지의 세로 길이(h)에 맞춰 정사각형으로 확대
                    cropped_resized = cv2.resize(cropped, (h, h), interpolation=cv2.INTER_NEAREST)
                    cropped_norm = cv2.normalize(cropped_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    cropped_color = cv2.applyColorMap(cropped_norm, cv2.COLORMAP_JET)
                    
                    # 가로로 이어 붙이기
                    combined_img = cv2.hconcat([depth_color, cropped_color])

                    # 4. 출력
                    cv2.imshow('Original (Left) vs Cropped (Right)', combined_img)
                    # q를 누르면 디버깅 종료하고 쫙 뽑고 싶다면 아래처럼 변경하셔도 좋습니다.
                    key = cv2.waitKey(0) 

                    # 데이터 저장 로직
                    tmp_imgs.append(cropped)
                    tmp_labels.append(label)
                    tmp_z.append(center[2])
                    if len(tmp_imgs) >= batch_size:
                        flush_to_zarr(img_ds, label_ds, z_ds)
                        
        flush_to_zarr(img_ds, label_ds, z_ds)

print(f"Zarr 데이터셋 생성 완료! 경로: {zarr_path}")