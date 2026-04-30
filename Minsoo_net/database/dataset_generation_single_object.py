from pathlib import Path

import numpy as np
import cv2
import trimesh
import random

from Minsoo_net.grasp.rendering import GraspRenderer
from Minsoo_net.grasp import GraspPipeline

# --- 설정 ---
use_visual=True

num_grasps = 10
quality_threshold = 0.002
max_angle = 30  # max_angle_deg
zarr_path = "grasp_dataset.zarr"
camera_offsets = np.linspace(0.45, 0.6, 10)  # cam_offset: start, stop, num 값 채워넣기


output_size = 32
crop_size=96

mesh_path=Path(__file__).parent.parent.resolve()
mesh_path=mesh_path/"data"/"object" / "Frankapanda" / "4"
mesh_files = list(mesh_path.rglob("*.obj")) + list(mesh_path.rglob("*.stl"))
random.shuffle(mesh_files)

def get_camera_positions(grasp, pose, offsets=[0.2, 0.25, 0.3]):
    T_world_gripper = pose @ grasp.T_grasp_obj
    z_axis = T_world_gripper[:3, 2]
    gripper_pos = T_world_gripper[:3, 3]
    cam_target = gripper_pos

    cam_positions = [gripper_pos - z_axis * d for d in offsets]
    return cam_positions, cam_target

for mesh_path in mesh_files:
    object_name=mesh_path.stem
    mesh_path=str(mesh_path)
    m = trimesh.load(mesh_path, force='mesh')
    if min(m.bounding_box_oriented.extents) < 0.005:
        print('너무 얇아서 제외')
        continue
    print(f'{object_name} 로드중')

    grasp_pipeline=GraspPipeline(mesh_path,quality_threshold=quality_threshold,num_grasps=num_grasps,max_approach_angle_deg=max_angle)
    renderer=GraspRenderer(mesh_path)
    if use_visual:
        viewer=renderer.scene.create_viewer()
        for _ in range(10):
            viewer.render()

    # --- 메인 루프 ---
    for pose, failed_grasps, quality_grasps, quality_scores in grasp_pipeline.execute(use_visual=True):
        renderer.set_stable_pose(pose)
        # 성공(quality) / 실패(0) 데이터 분류
        tasks = [(quality_grasps, quality_scores), 
                (failed_grasps, [0.0]*len(failed_grasps))]

        for grasps, labels in tasks:
            print("이미지 랜더링 시작 (진행중..)")
            for grasp, label in zip(grasps, labels):
                cam_poses,cam_target=get_camera_positions(grasp,pose,offsets=camera_offsets)
                metalic,roughness=renderer.sample_material()
                renderer.set_material(metalic=metalic,roughness=roughness)
                for cam_pos in cam_poses:
                    depth = renderer.render(camera_pos=cam_pos,target_pos=cam_target)
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
                        renderer.scene.step()
                        renderer.scene.update_render()
                        renderer.sensor.take_picture()
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
                        cv2.waitKey(0)

print(f"Zarr 데이터셋 생성 완료! 경로: {zarr_path}")