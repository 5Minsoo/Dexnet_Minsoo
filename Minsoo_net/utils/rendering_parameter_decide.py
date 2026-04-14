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

path='/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/object/FKT38-A50-P25.stl'
# --- 환경 설정 및 객체 로드 ---
mesh = trimesh.load(path)
stable_poses, stable_probs = mesh.compute_stable_poses()
num_stable = len(stable_poses)
print(f"\n총 {num_stable}개의 stable pose 발견")
for i, prob in enumerate(stable_probs):
    print(f"  Pose {i}: 확률 {prob:.4f}")

def apply_stable_pose(pose_idx):
    SE = stable_poses[pose_idx]
    rot = R.from_matrix(SE[:3,:3]).as_quat()
    q = [rot[3], rot[0], rot[1], rot[2]]
    t = SE[:3,3] * 0.001
    return t, q

t, r_quat = apply_stable_pose(0)

scene = sapien.Scene()
scene.set_timestep(1)
scene.add_ground(altitude=0)
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
sapien.render.set_camera_shader_dir("rt")
sapien.render.set_ray_tracing_samples_per_pixel(32)
sapien.render.set_ray_tracing_path_depth(8)
sapien.render.set_ray_tracing_denoiser("optix")

material = sapien.render.RenderMaterial()
material.set_metallic(1.0)
material.set_roughness(0.6)

builder = scene.create_actor_builder()
builder.add_visual_from_file(path, scale=(0.001,0.001,0.001), material=material)
bin_obj = builder.build_kinematic(name="bin")
bin_obj.set_pose(sapien.Pose(p=t, q=r_quat))

# --- 센서 설정 ---
cam_pos = np.array([0.0, 0.0, 0.5])
orientation = look_at(cam_pos)

sensor_config = StereoDepthSensorConfig(model="D435")
sensor_mount_actor = scene.create_actor_builder().build_kinematic()
sensor = StereoDepthSensor(config=sensor_config, mount_entity=sensor_mount_actor, pose=Pose(cam_pos, orientation))

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
cv2.createTrackbar("Cam Theta", "Depth Viewer", 0, 180, nothing)
cv2.createTrackbar("Cam Phi", "Depth Viewer", 180, 360, nothing)
cv2.createTrackbar("Stable Pose", "Depth Viewer", 0, max(num_stable - 1, 0), nothing)

prev_sensor_params = {'bw': 2, 'unq': 50, 'p1': 8, 'p2': 24}
prev_cam_params = {'r': 50, 'theta': 0, 'phi': 180}
prev_pose_idx = 0
real_bw = 7
real_p2 = 24

viewer = scene.create_viewer()
viewer.set_camera_pose(sensor.get_pose())

print("\n--- 조작 안내 ---")
print("Stable Pose 슬라이더: stable pose 전환")
print("'s' 키: 파라미터 저장 / 'q' 키: 종료\n")

while not viewer.closed:
    # --- 1. 머티리얼 ---
    metallic_val = cv2.getTrackbarPos("Metallic", "Depth Viewer") / 100.0
    roughness_val = cv2.getTrackbarPos("Roughness", "Depth Viewer") / 100.0
    material.set_metallic(metallic_val)
    material.set_roughness(roughness_val)

    # --- 2. Stable Pose 토글 ---
    cur_pose_idx = cv2.getTrackbarPos("Stable Pose", "Depth Viewer")
    if cur_pose_idx != prev_pose_idx:
        t, r_quat = apply_stable_pose(cur_pose_idx)
        bin_obj.set_pose(sapien.Pose(p=t, q=r_quat))
        prev_pose_idx = cur_pose_idx
        print(f"Stable Pose {cur_pose_idx} 적용 (확률: {stable_probs[cur_pose_idx]:.4f})")

    # --- 3. 카메라 ---
    cur_cam_r = cv2.getTrackbarPos("Cam R (x100)", "Depth Viewer")
    cur_cam_theta = cv2.getTrackbarPos("Cam Theta", "Depth Viewer")
    cur_cam_phi = cv2.getTrackbarPos("Cam Phi", "Depth Viewer")

    cam_r = max(cur_cam_r, 1) / 100.0
    cam_theta = np.radians(cur_cam_theta)
    cam_phi = np.radians(cur_cam_phi - 180)

    cam_pos = np.array([
        cam_r * np.sin(cam_theta) * np.cos(cam_phi),
        cam_r * np.sin(cam_theta) * np.sin(cam_phi),
        cam_r * np.cos(cam_theta)
    ])
    orientation = look_at(cam_pos)

    cam_changed = (cur_cam_r != prev_cam_params['r'] or
                   cur_cam_theta != prev_cam_params['theta'] or
                   cur_cam_phi != prev_cam_params['phi'])

    # --- 4. 센서 ---
    cur_bw_val = cv2.getTrackbarPos("Block Width (Odd)", "Depth Viewer")
    cur_unq = cv2.getTrackbarPos("Uniqueness", "Depth Viewer")
    cur_p1 = cv2.getTrackbarPos("P1 Penalty", "Depth Viewer")
    cur_p2 = cv2.getTrackbarPos("P2 Penalty", "Depth Viewer")

    sensor_changed = (cur_bw_val != prev_sensor_params['bw'] or
                      cur_unq != prev_sensor_params['unq'] or
                      cur_p1 != prev_sensor_params['p1'] or
                      cur_p2 != prev_sensor_params['p2'])

    if sensor_changed or cam_changed:
        scene.remove_actor(sensor_mount_actor)
        sensor_mount_actor = scene.create_actor_builder().build_kinematic()

        real_bw = max(1, cur_bw_val) * 2 + 1
        real_p2 = max(cur_p1 + 1, cur_p2)

        sensor_config.block_width = real_bw
        sensor_config.block_height = real_bw
        sensor_config.uniqueness_ratio = cur_unq
        sensor_config.p1_penalty = cur_p1
        sensor_config.p2_penalty = real_p2

        sensor = StereoDepthSensor(config=sensor_config, mount_entity=sensor_mount_actor, pose=Pose(cam_pos, orientation))
        viewer.set_camera_pose(sensor.get_pose())

        prev_sensor_params = {'bw': cur_bw_val, 'unq': cur_unq, 'p1': cur_p1, 'p2': cur_p2}
        prev_cam_params = {'r': cur_cam_r, 'theta': cur_cam_theta, 'phi': cur_cam_phi}

    # --- 5. 렌더링 ---
    scene.update_render()
    viewer.render()
    sensor.take_picture()
    sensor.compute_depth()

    # --- 6. Raw Depth 시각화 ---
    depth = sensor.get_depth()
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_gray = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)

    # --- 7. OSD ---
    info_texts = [
        f"[Material] Metallic: {metallic_val:.2f}, Roughness: {roughness_val:.2f}",
        f"[Sensor] BW: {real_bw}, Uniq: {sensor_config.uniqueness_ratio}, P1: {sensor_config.p1_penalty}, P2: {sensor_config.p2_penalty}",
        f"[Camera] R: {cam_r:.2f}, Theta: {np.degrees(cam_theta):.0f}, Phi: {np.degrees(cam_phi):.0f}",
        f"[Stable Pose] {cur_pose_idx}/{num_stable-1} (prob: {stable_probs[cur_pose_idx]:.4f})",
    ]

    y0, dy = 30, 30
    for i, text in enumerate(info_texts):
        y = y0 + i * dy
        cv2.putText(depth_gray, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(depth_gray, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.imshow("Depth Viewer", depth_gray)

    # --- 8. 키보드 ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        save_data = {
            "material": {
                "metallic": float(metallic_val),
                "roughness": float(roughness_val)
            },
            "sensor_config": {
                "block_width": real_bw,
                "uniqueness_ratio": sensor_config.uniqueness_ratio,
                "p1_penalty": sensor_config.p1_penalty,
                "p2_penalty": sensor_config.p2_penalty
            },
            "camera": {
                "r": float(cam_r),
                "theta_deg": float(np.degrees(cam_theta)),
                "phi_deg": float(np.degrees(cam_phi)),
            },
            "stable_pose": {
                "index": int(cur_pose_idx),
                "probability": float(stable_probs[cur_pose_idx]),
                "transform": stable_poses[cur_pose_idx].tolist()
            }
        }
        with open("saved_params.json", "w") as f:
            json.dump(save_data, f, indent=4)
        print("✅ 파라미터 저장 완료!")

    scene.step()

cv2.destroyAllWindows()