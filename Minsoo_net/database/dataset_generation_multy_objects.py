from Minsoo_net.grasp.rendering import GraspRenderer
from Minsoo_net.grasp import GraspPipeline
import numpy as np
import zarr, sys
import cv2
from pathlib import Path
import time
# --- 설정 ---
use_visual=False

num_grasps=300
quality_threshold=0.002
prob_threshold=0.012 
num_poses=10
max_angle=15

num_camera_points=200
camera_radius=(0.40,0.60)
camera_tilt=(0,3.14/6)


zarr_path = "grasp_dataset.zarr"
output_size = 32
crop_size=96
batch_size = 10 

print("다시쓰기 / 이어하기 선택")
mode=input("모드 선택(처음부터: 1번 / 이어하기 2번) 입력 후 Enter ")


if mode=="1":
    print("기존 데이터를 삭제하고 새로 작업을 시작합니다.")
    store = zarr.open(zarr_path, mode='w')
elif mode==2:
    print("기존 데이터에 이어서 작업을 시작합니다.")
    store = zarr.open(zarr_path, mode='a')
else:
    sys.exit()

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


for mesh_path in mesh_files:
    object_name=mesh_path.stem
    mesh_path=str(mesh_path)
    print(f'{object_name} 로드중')

    grasp_pipeline=GraspPipeline(mesh_path,quality_threshold=quality_threshold,num_grasps=num_grasps,max_approach_angle_deg=max_angle,num_poses=num_poses)
    renderer=GraspRenderer(mesh_path)
    points=GraspRenderer.sample_spherical_positions(camera_radius,camera_tilt,(0,3.14),num_points=num_camera_points)
    if use_visual:
        viewer=renderer.scene.create_viewer()
        for _ in range(10):
            viewer.render()
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


    finish=time.time()
    # --- 메인 루프 ---
    for pose, failed_grasps, quality_grasps, quality_scores in grasp_pipeline.execute(start_index=start_idx):
        start=time.time()
        print(f'Grasp sampling 시간 {int(start-finish)}초')
        renderer.set_stable_pose(pose)
        metalic,roughness=renderer.sample_material(num_camera_points)
        pose_group=obj_group.require_group(f"pose{start_idx}")
        start_idx+=1
        img_ds = pose_group.create_array("images", shape=(0, output_size, output_size), 
                                chunks=(batch_size, output_size, output_size), 
                                dtype='float32',overwrite=True)
        label_ds = pose_group.create_array("labels", shape=(0,), 
                                        chunks=(batch_size,), 
                                        dtype='float32',overwrite=True)
        z_ds = pose_group.create_array("gripper_depth", shape=(0,), 
                                        chunks=(batch_size,), 
                                        dtype='float32',overwrite=True)
        # 성공(quality) / 실패(0) 데이터 분류
        tasks = [(quality_grasps, quality_scores), 
                (failed_grasps, [0.0]*len(failed_grasps))]

        for grasps, labels in tasks:
            print("이미지 랜더링 시작 (진행중..)")
            for grasp, label in zip(grasps, labels):
                metalic,roughness=renderer.sample_material()
                renderer.set_material(metalic=metalic[0],roughness=roughness[0])
                for point in points:
                    depth = renderer.render(camera_pos=point)
                    # 좌표 계산 및 시각화 (기존 로직)
                    origin = [0, 0, 0]
                    center = (pose @ np.append(grasp.center, 1.0))[:3]
                    axis = (pose @ np.append(grasp.axis, 1.0))[:3]
                    image_point = renderer.world_to_pixel([origin, center, axis])
                    grasp_depth=(renderer.get_extrinsic()@np.append(center,1.0))[2]
                    
                    cropped = GraspRenderer.crop_grasp_image(
                        depth, image_point[1], image_point[2]-image_point[0], 
                        crop_size=crop_size, output_size=output_size
                    )
                    if use_visual:
                        viewer.render()
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
                        viewer.render()
                        # 디버깅 시 하나씩 확인하려면 waitKey(0), 자동으로 휙휙 넘어가게 하려면 waitKey(1)
                        cv2.waitKey(1)
                    
                    tmp_imgs.append(cropped)
                    tmp_labels.append(label)
                    tmp_z.append(grasp_depth)
                    if len(tmp_imgs) >= batch_size:
                        flush_to_zarr(img_ds,label_ds,z_ds)
        flush_to_zarr(img_ds,label_ds,z_ds)
        finish=time.time()
        print(f'이미지 랜더링 종료 Pose{start_idx-1} 걸린시간: {int(finish-start)}초')


print(f"Zarr 데이터셋 생성 완료! 경로: {zarr_path}")