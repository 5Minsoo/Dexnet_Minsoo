import sapien.core as sapien
import numpy as np
import cv2, math
import trimesh
import json
from sapien.core import Pose
from sapien.sensor import StereoDepthSensor, StereoDepthSensorConfig
from scipy.spatial.transform import Rotation as R

sapien.set_log_level("error")

# --- 헬퍼 함수들 ---
def look_at(camera_pos, target_point=None, up_vector=np.array([0.0,0.0,1.0])):
    x = -camera_pos
    x_norm = x/np.linalg.norm(x)
    right = np.cross(camera_pos,up_vector)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([0.0,1.0,0.0])
    right_norm = right/np.linalg.norm(right)
    up = np.cross(x_norm,right_norm)
    se = np.c_[x_norm,right_norm,up]
    rot = R.from_matrix(se).as_quat()
    return [rot[3],rot[0],rot[1],rot[2]]

path='/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/object/example_model.obj'
cam_mesh_path = '/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/object/1/00001147_274866da478e4a009bed8750_step_002_0012.obj'

# --- 환경 설정 및 객체 로드 ---
mesh = trimesh.load(path)
stable_poses, stable_probs = mesh.compute_stable_poses()
num_stable = len(stable_poses)
print(f"\n총 {num_stable}개의 stable pose 발견")

def apply_stable_pose(pose_idx):
    SE = stable_poses[pose_idx]
    rot = R.from_matrix(SE[:3,:3]).as_quat()
    q = [rot[3], rot[0], rot[1], rot[2]]
    t = SE[:3,3]
    return t, q

t, r_quat = apply_stable_pose(0)

scene = sapien.Scene()
ground_material = sapien.render.RenderMaterial()
ground_material.set_base_color([0.9, 0.45, 0.1, 1.0])

scene.set_timestep(1)
scene.add_ground(altitude=0, render_material=ground_material)
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
sapien.render.set_camera_shader_dir("rt")
sapien.render.set_ray_tracing_samples_per_pixel(32)
sapien.render.set_ray_tracing_path_depth(8)
sapien.render.set_ray_tracing_denoiser("optix")

material = sapien.render.RenderMaterial()
material.set_metallic(1.0)
material.set_roughness(0.6)
material.set_base_color([0.6, 0.6, 0.6, 1.0]) 

builder = scene.create_actor_builder()
builder.add_visual_from_file(path, material=material)
bin_obj = builder.build_kinematic(name="bin")
bin_obj.set_pose(sapien.Pose(p=t, q=r_quat))

# --- 다중 카메라 Mesh 액터 생성 (눈에 보이는 것 7개) ---
cam_material = sapien.render.RenderMaterial()
cam_material.set_metallic(0.3)
cam_material.set_roughness(0.7)
cam_material.set_base_color([0.2, 0.2, 0.8, 1.0]) # 눈에 잘 띄게 파란색 계열로 변경

fan_angles = [-45, -30, -15, 0, 15, 30, 45]
cam_actors = []

for i, angle in enumerate(fan_angles):
    cam_builder = scene.create_actor_builder()
    cam_builder.add_visual_from_file(
        filename=cam_mesh_path,
        scale=[0.001, 0.001, 0.001],
        material=cam_material,
    )
    cam_actors.append(cam_builder.build_kinematic(name=f"d455_cam_{angle}deg"))

# --- 실제 뎁스를 찍는 센서 (에러 방지를 위해 중앙에 1개만) ---
sensor_config = StereoDepthSensorConfig(model="D435")
sensor_mount = scene.create_actor_builder().build_kinematic(name="main_sensor_mount")
sensor = StereoDepthSensor(config=sensor_config, mount_entity=sensor_mount)

# --- OpenCV 슬라이더 ---
def nothing(x): pass

cv2.namedWindow("Depth Viewer", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Depth Viewer", 800, 800)

cv2.createTrackbar("Metallic", "Depth Viewer", 100, 100, nothing)
cv2.createTrackbar("Roughness", "Depth Viewer", 60, 100, nothing)
cv2.createTrackbar("Block Width (Odd)", "Depth Viewer", 2, 15, nothing)
cv2.createTrackbar("Uniqueness", "Depth Viewer", 50, 100, nothing)
cv2.createTrackbar("P1 Penalty", "Depth Viewer", 8, 100, nothing)
cv2.createTrackbar("P2 Penalty", "Depth Viewer", 24, 200, nothing)
cv2.createTrackbar("Cam R (x100)", "Depth Viewer", 50, 200, nothing)
# 핵심 수정: 카메라가 겹치지 않게 초기 Theta 값을 45도로 설정
cv2.createTrackbar("Cam Theta", "Depth Viewer", 45, 180, nothing) 
cv2.createTrackbar("Cam Phi", "Depth Viewer", 180, 360, nothing)
cv2.createTrackbar("Stable Pose", "Depth Viewer", 0, max(num_stable - 1, 0), nothing)

prev_sensor_params = {'bw': -1, 'unq': -1, 'p1': -1, 'p2': -1}
prev_cam_params = {'r': -1, 'theta': -1, 'phi': -1}
prev_pose_idx = 0
real_bw = 7
real_p2 = 24

viewer = scene.create_viewer()

while not viewer.closed:
    metallic_val = cv2.getTrackbarPos("Metallic", "Depth Viewer") / 100.0
    roughness_val = cv2.getTrackbarPos("Roughness", "Depth Viewer") / 100.0
    material.set_metallic(metallic_val)
    material.set_roughness(roughness_val)

    cur_pose_idx = cv2.getTrackbarPos("Stable Pose", "Depth Viewer")
    if cur_pose_idx != prev_pose_idx:
        t, r_quat = apply_stable_pose(cur_pose_idx)
        bin_obj.set_pose(sapien.Pose(p=t, q=r_quat))
        prev_pose_idx = cur_pose_idx

    cur_cam_r = cv2.getTrackbarPos("Cam R (x100)", "Depth Viewer")
    cur_cam_theta = cv2.getTrackbarPos("Cam Theta", "Depth Viewer")
    cur_cam_phi = cv2.getTrackbarPos("Cam Phi", "Depth Viewer")

    cur_bw_val = cv2.getTrackbarPos("Block Width (Odd)", "Depth Viewer")
    cur_unq = cv2.getTrackbarPos("Uniqueness", "Depth Viewer")
    cur_p1 = cv2.getTrackbarPos("P1 Penalty", "Depth Viewer")
    cur_p2 = cv2.getTrackbarPos("P2 Penalty", "Depth Viewer")

    cam_changed = (cur_cam_r != prev_cam_params['r'] or
                   cur_cam_theta != prev_cam_params['theta'] or
                   cur_cam_phi != prev_cam_params['phi'])

    sensor_changed = (cur_bw_val != prev_sensor_params['bw'] or
                      cur_unq != prev_sensor_params['unq'] or
                      cur_p1 != prev_sensor_params['p1'] or
                      cur_p2 != prev_sensor_params['p2'])

    if sensor_changed or cam_changed:
        cam_r = max(cur_cam_r, 1) / 100.0
        cam_theta = np.radians(cur_cam_theta)
        base_phi = np.radians(cur_cam_phi - 180)

        if sensor_changed:
            scene.remove_actor(sensor_mount)
            sensor_mount = scene.create_actor_builder().build_kinematic(name="main_sensor_mount")
            
            real_bw = max(1, cur_bw_val) * 2 + 1
            real_p2 = max(cur_p1 + 1, cur_p2)
            
            sensor_config.block_width = real_bw
            sensor_config.block_height = real_bw
            sensor_config.uniqueness_ratio = cur_unq
            sensor_config.p1_penalty = cur_p1
            sensor_config.p2_penalty = real_p2
            
            sensor = StereoDepthSensor(config=sensor_config, mount_entity=sensor_mount)

        center_pose = None

        # 7개 카메라 메쉬를 부채꼴 모양으로 재배치
        for i, angle_offset in enumerate(fan_angles):
            # 중심 각도(base_phi)를 기준으로 15도씩 벌어짐
            phi = base_phi + np.radians(angle_offset)

            cam_pos = np.array([
                cam_r * np.sin(cam_theta) * np.cos(phi),
                cam_r * np.sin(cam_theta) * np.sin(phi),
                cam_r * np.cos(cam_theta)
            ])
            orientation = look_at(cam_pos)

            quat_xyzw = [orientation[1], orientation[2], orientation[3], orientation[0]]
            rot_mesh = R.from_quat(quat_xyzw) * R.from_euler('y', 90, degrees=True) * R.from_euler('z', 90, degrees=True)
            q = rot_mesh.as_quat()
            orientation_mesh = [q[3], q[0], q[1], q[2]]

            cam_actors[i].set_pose(Pose(cam_pos, orientation_mesh))

            # 중앙(0도) 카메라일 때 실제 센서 위치 업데이트
            if angle_offset == 0:
                center_pose = Pose(cam_pos, orientation)
                sensor_mount.set_pose(center_pose)

        # 뷰어 카메라도 중앙 센서 위치로 업데이트
        if center_pose:
            viewer.set_camera_pose(center_pose)

        prev_sensor_params = {'bw': cur_bw_val, 'unq': cur_unq, 'p1': cur_p1, 'p2': cur_p2}
        prev_cam_params = {'r': cur_cam_r, 'theta': cur_cam_theta, 'phi': cur_cam_phi}

    scene.update_render()
    viewer.render()

    sensor.take_picture()
    sensor.compute_depth()

    depth = sensor.get_depth()
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_gray = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)

    info_texts = [
        # f"[Material] Metallic: {metallic_val:.2f}, Roughness: {roughness_val:.2f}",
        # f"[Sensor] BW: {real_bw}, Uniq: {sensor_config.uniqueness_ratio}, P1: {sensor_config.p1_penalty}, P2: {sensor_config.p2_penalty}",
        # f"[Camera] R: {cam_r:.2f}, Theta: {np.degrees(cam_theta):.0f}, Phi: {cur_cam_phi-180}",
        # f"[Stable Pose] {cur_pose_idx}/{num_stable-1}",
        # "[Info] 7 Meshes spawned in a fan shape."
    ]

    y0, dy = 30, 30
    for i, text in enumerate(info_texts):
        y = y0 + i * dy
        cv2.putText(depth_gray, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(depth_gray, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.imshow("Depth Viewer", depth_gray)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    scene.step()

cv2.destroyAllWindows()