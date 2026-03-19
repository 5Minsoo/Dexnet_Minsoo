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
def generate_spherical_points(r_range, theta_range, phi_range, num_points=1000):
    u, v, w = np.random.rand(3, num_points)
    r = (r_range[0]**3 + u * (r_range[1]**3 - r_range[0]**3))**(1/3)
    cos_t_min, cos_t_max = np.cos(theta_range[0]), np.cos(theta_range[1])
    theta = np.arccos(cos_t_min + v * (cos_t_max - cos_t_min))
    phi = phi_range[0] + w * (phi_range[1] - phi_range[0])
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=1)

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

def align_crop_and_scale(image, center, axis):
    dx, dy = axis
    angle_radians = math.atan2(dy, dx) 
    angle_degrees = math.degrees(angle_radians)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, scale=1.0)
    h, w = image.shape[:2]
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    crop_size = (256, 256)
    cropped_image = cv2.getRectSubPix(rotated_image, crop_size, center)
    final_image = cv2.resize(cropped_image, (32, 32), interpolation=cv2.INTER_AREA)
    return final_image

def draw_grasp_debug(image, center, axis, line_length=200):
    debug_image = image.copy()
    cx, cy = int(center[0]), int(center[1])
    cv2.circle(debug_image, (cx, cy), radius=5, color=(0, 0, 255), thickness=-1)
    dx, dy = axis
    magnitude = math.hypot(dx, dy)
    if magnitude > 0:
        ux = dx / magnitude
        uy = dy / magnitude
        pt1 = (int(cx - ux * line_length), int(cy - uy * line_length))
        pt2 = (int(cx + ux * line_length), int(cy + uy * line_length))
        cv2.line(debug_image, pt1, pt2, color=(0, 255, 0), thickness=2)
    return debug_image

# --- 환경 설정 및 객체 로드 ---
mesh = trimesh.load('/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/object/object.stl')
trans, _ = mesh.compute_stable_poses()
trans = trans[1]
r_quat = R.from_matrix(trans[:3,:3]).as_quat()
r_quat = np.array([r_quat[3], r_quat[0], r_quat[1], r_quat[2]]) 
t = trans[:3,3] * 0.001

scene = sapien.Scene()
scene.set_timestep(1)
scene.add_ground(altitude=0)
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
sapien.render.set_camera_shader_dir("rt")
sapien.render.set_ray_tracing_samples_per_pixel(32)
sapien.render.set_ray_tracing_path_depth(8)
sapien.render.set_ray_tracing_denoiser("optix")

# 머티리얼 초기 설정
material = sapien.render.RenderMaterial()
material.set_metallic(1.0)
material.set_roughness(0.6)

builder = scene.create_actor_builder()
builder.add_visual_from_file('/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/object/object.stl', scale=(0.001,0.001,0.001), material=material)
bin = builder.build_kinematic(name="bin")
bin.set_pose(sapien.Pose(p=t, q=r_quat))

# --- 기본 센서 및 고정 카메라 설정 ---
# 기존 슬라이더의 기본값을 사용해 카메라 위치 고정
fixed_r = 0.5
fixed_theta = np.radians(45)
fixed_phi = np.radians(45)

cam_pos = np.array([
    fixed_r * np.sin(fixed_theta) * np.cos(fixed_phi),
    fixed_r * np.sin(fixed_theta) * np.sin(fixed_phi),
    fixed_r * np.cos(fixed_theta)
])
orientation = look_at(cam_pos)

sensor_config = StereoDepthSensorConfig(model="D435")
sensor_mount_actor = scene.create_actor_builder().build_kinematic()
sensor = StereoDepthSensor(config=sensor_config, mount_entity=sensor_mount_actor, pose=Pose(cam_pos, orientation))

R_S2O = np.array([
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [1,  0,  0, 0],
    [0,  0,  0, 1]
])
intrinsic = np.eye(4)
intrinsic[:3,:3] = sensor_config.ir_intrinsic

# --- OpenCV 창 및 슬라이더(Trackbar) 설정 ---
def nothing(x): pass

cv2.namedWindow("Depth Viewer", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Depth Viewer", 800, 800) 

# 머티리얼 파라미터 (0~100 범위를 0.0~1.0으로 사용)
cv2.createTrackbar("Metallic", "Depth Viewer", 100, 100, nothing) 
cv2.createTrackbar("Roughness", "Depth Viewer", 60, 100, nothing)

# 센서 파라미터
cv2.createTrackbar("Block Width (Odd)", "Depth Viewer", 2, 15, nothing) 
cv2.createTrackbar("Uniqueness", "Depth Viewer", 50, 100, nothing)
cv2.createTrackbar("P1 Penalty", "Depth Viewer", 8, 100, nothing)
cv2.createTrackbar("P2 Penalty", "Depth Viewer", 24, 200, nothing)

prev_sensor_params = {'bw': 2, 'unq': 50, 'p1': 8, 'p2': 24}
real_bw = 7 
real_p2 = 24

viewer = scene.create_viewer()
viewer.set_camera_pose(sensor.get_pose()) # 초기 카메라 뷰어 동기화

print("\n--- 조작 안내 ---")
print("슬라이더를 움직여 재질(Material)과 센서 파라미터를 조절하세요.")
print("'s' 키: 현재 파라미터 JSON 파일로 저장")
print("'q' 키: 프로그램 종료\n")

while not viewer.closed:
    # --- 1. 머티리얼 업데이트 ---
    # 슬라이더 값(0~100)을 0.0~1.0 범위로 변환하여 적용
    metallic_val = cv2.getTrackbarPos("Metallic", "Depth Viewer") / 100.0
    roughness_val = cv2.getTrackbarPos("Roughness", "Depth Viewer") / 100.0
    material.set_metallic(metallic_val)
    material.set_roughness(roughness_val)

    # --- 2. 센서 Config 업데이트 ---
    cur_bw_val = cv2.getTrackbarPos("Block Width (Odd)", "Depth Viewer")
    cur_unq = cv2.getTrackbarPos("Uniqueness", "Depth Viewer")
    cur_p1 = cv2.getTrackbarPos("P1 Penalty", "Depth Viewer")
    cur_p2 = cv2.getTrackbarPos("P2 Penalty", "Depth Viewer")

    if (cur_bw_val != prev_sensor_params['bw'] or 
        cur_unq != prev_sensor_params['unq'] or 
        cur_p1 != prev_sensor_params['p1'] or 
        cur_p2 != prev_sensor_params['p2']):
        
        # 조명 누적(Too many textured lights) 방지를 위해 기존 액터 제거
        scene.remove_actor(sensor_mount_actor)
        sensor_mount_actor = scene.create_actor_builder().build_kinematic()
        
        real_bw = max(1, cur_bw_val) * 2 + 1 
        real_p2 = max(cur_p1 + 1, cur_p2) 

        sensor_config.block_width = real_bw
        sensor_config.block_height = real_bw
        sensor_config.uniqueness_ratio = cur_unq
        sensor_config.p1_penalty = cur_p1
        sensor_config.p2_penalty = real_p2

        # 센서 재생성 (고정된 cam_pos, orientation 사용)
        sensor = StereoDepthSensor(config=sensor_config, mount_entity=sensor_mount_actor, pose=Pose(cam_pos, orientation))
        viewer.set_camera_pose(sensor.get_pose()) # 센서 재생성 시 뷰어 포즈 재동기화
        
        prev_sensor_params = {'bw': cur_bw_val, 'unq': cur_unq, 'p1': cur_p1, 'p2': cur_p2}

    # --- 3. 씬 렌더링 및 뎁스 연산 ---
    scene.update_render()
    viewer.render()
    sensor.take_picture()
    sensor.compute_depth()
    
    # --- 4. 이미지 투영 및 시각화 ---
    extrinsic = R_S2O @ np.linalg.inv(sensor.get_pose().to_transformation_matrix())
    world_point = np.array([0, 0, 0, 1])
    image_point = intrinsic @ extrinsic @ world_point
    
    depth = sensor.get_depth()
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    if image_point[2] > 0:
        u = int(image_point[0] / image_point[2])
        v = int(image_point[1] / image_point[2])
        if 0 <= u < depth_colored.shape[1] and 0 <= v < depth_colored.shape[0]:
            cv2.circle(depth_colored, (u, v), 5, (0, 0, 0), -1)

    # --- 5. 이미지 위에 파라미터 정보 텍스트 오버레이 (OSD) ---
    info_texts = [
        f"[Material] Metallic: {metallic_val:.2f}, Roughness: {roughness_val:.2f}",
        f"[Sensor] Block Width: {real_bw}",
        f"[Sensor] Uniqueness: {sensor_config.uniqueness_ratio}",
        f"[Sensor] P1 Penalty: {sensor_config.p1_penalty}",
        f"[Sensor] P2 Penalty: {sensor_config.p2_penalty}"
    ]
    
    y0, dy = 30, 30
    for i, text in enumerate(info_texts):
        y = y0 + i * dy
        # 가독성을 위해 검은색 테두리를 먼저 그리고 하얀색 글씨 작성
        cv2.putText(depth_colored, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(depth_colored, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.imshow("Depth Viewer", depth_colored)
    
    # --- 6. 키보드 입력 처리 ('s' 저장, 'q' 종료) ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord('s'):
        # 현재 파라미터를 JSON 형태로 묶어서 파일로 저장
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
            }
        }
        with open("saved_params.json", "w") as f:
            json.dump(save_data, f, indent=4)
        print("✅ [성공] 현재 파라미터가 'saved_params.json' 파일에 저장되었습니다!")

    scene.step()

cv2.destroyAllWindows()