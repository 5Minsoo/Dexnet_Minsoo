import sapien
import sapien.core as sapien_core
import numpy as np
import cv2
import math
from sapien.core import Pose
from sapien.sensor import StereoDepthSensor, StereoDepthSensorConfig
from Minsoo_net.grasp.random_variables import ParamsGaussianRV
from scipy.spatial.transform import Rotation as R
import yaml,trimesh
class GraspRenderer:
    """
    DexNet용 깊이 이미지 렌더러.
    
    물체를 stable pose로 배치한 뒤, 구면 좌표계 상의 다양한 시점에서
    depth 이미지를 렌더링하고, grasp 중심점/축 기반으로 크롭된 학습 데이터를 생성한다.
    
    Usage:
        renderer = GraspRenderer("path/to/object.stl")
        renderer.set_stable_pose(rotation_matrix)  # stable pose 적용
        
        # 단일 시점 렌더링
        depth = renderer.render(camera_pos)
        
        # 구면 좌표 랜덤 시점 배치 렌더링
        for depth, extrinsic in renderer.render_spherical(num_views=100):
            ...
        
        # DexNet grasp 후보에서 크롭 이미지 생성
        cropped = renderer.crop_grasp_image(depth, center_px, axis_px)
    """

    # ── 초기화 ──────────────────────────────────────────────

    _render_initialized = False

    def __init__(
        self,
        mesh_path: str,
        mesh_scale: tuple = (0.001, 0.001, 0.001),
        sensor_model: str = "D435",
        image_size: tuple = (480, 640),      # (H, W) — D435 기본
        spp: int = 32,
        path_depth: int = 8,
        config_path='/home/minsoo/Dexnet_Minsoo/Minsoo_net/config/master_config.yaml'
    ):
        self.mesh_path = mesh_path
        self.mesh_scale = mesh_scale
        sapien.set_log_level("error")
        # ── 전역 렌더 설정 (프로세스당 1회만) ──
        if not GraspRenderer._render_initialized:
            sapien.render.set_camera_shader_dir("rt")
            sapien.render.set_ray_tracing_samples_per_pixel(spp)
            sapien.render.set_ray_tracing_path_depth(path_depth)
            sapien.render.set_ray_tracing_denoiser("optix")
            GraspRenderer._render_initialized = True

        self.rv=ParamsGaussianRV(config_yaml=config_path)
        self.material=sapien.render.RenderMaterial()
        self.material.set_metallic(1.0)
        self.material.set_roughness(0.8)

        # ── Scene ──
        self.scene = sapien.Scene()
        self.scene.set_timestep(1 / 1)
        self.scene.add_ground(altitude=0)
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
        
        # ── 물체 로드 ──
        builder = self.scene.create_actor_builder()
        builder.add_convex_collision_from_file(mesh_path)
        builder.add_visual_from_file(mesh_path, scale=(0.001,0.001,0.001),material=self.material)
        self.obj = builder.build_kinematic(name="object")
        self.obj.set_pose(Pose(p=[0, 0, 0]))

        ################################## ── 센서(카메라) ──###############################################
        self.sensor_config = StereoDepthSensorConfig(model=sensor_model)
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.sensor_config.block_height=config.get('block_height',7)
        self.sensor_config.block_width=self.sensor_config.block_height
        self.sensor_config.uniqueness_ratio=config.get('uniqueness_ratio',50)
        self._mount = self.scene.create_actor_builder().build_kinematic()
        # 초기 위치는 임시 — render() 호출 시 갱신
        self.sensor = StereoDepthSensor(
            config=self.sensor_config,
            mount_entity=self._mount,
            pose=Pose([0, 0, 0.3]),
        )

        # SAPIEN 센서→OpenCV 좌표 변환 (고정)
        self._R_S2O = np.array([
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [1,  0,  0, 0],
            [0,  0,  0, 1],
        ])

        # intrinsic (센서 생성 후 고정)
        self.intrinsic = np.eye(4)
        self.intrinsic[:3, :3] = self.sensor_config.ir_intrinsic

    def set_material(self,metalic,roughness):
        self.material.set_metallic(metalic)
        self.material.set_roughness(roughness)


    # ── Stable Pose ─────────────────────────────────────────

    def set_stable_pose(self, SE):
        """
        물체의 stable pose를 설정한다.
        
        :param rotation: (3,3) 회전 행렬 또는 (4,) 쿼터니언 [w,x,y,z]
        :param position: (3,) 위치. None이면 [0,0,0] 유지.
        """
        rotation=SE[:3,:3]
        q = R.from_matrix(rotation).as_quat()  # [x,y,z,w]
        q = [q[3], q[0], q[1], q[2]]           # SAPIEN 형식 [w,x,y,z]
        
        p=SE[:3,3]
        self.obj.set_pose(Pose(p=p, q=q))

    def sample_material(self,size=1):
        return self.rv.sample_material(size)

    # ── 카메라 유틸 ─────────────────────────────────────────

    @staticmethod
    def _look_at(camera_pos: np.ndarray, target: np.ndarray = np.zeros(3), up: np.ndarray = np.array([0, 0, 1.0])):
        """target을 바라보는 카메라 orientation 쿼터니언 [w,x,y,z] 반환."""
        forward = target - camera_pos
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        up_corrected = np.cross(forward, right)
        mat = np.column_stack([forward, right, up_corrected])
        q = R.from_matrix(mat).as_quat()  # [x,y,z,w]
        return [q[3], q[0], q[1], q[2]]

    @staticmethod
    def sample_spherical_positions(
        r_range: tuple,
        theta_range: tuple,
        phi_range: tuple,
        num_points: int = 100,
    ) -> np.ndarray:
        """구면 좌표계에서 공간 균일 분포 시점 샘플링. shape=(N,3)"""
        u, v, w = np.random.rand(3, num_points)

        r = (r_range[0]**3 + u * (r_range[1]**3 - r_range[0]**3)) ** (1 / 3)

        cos_min, cos_max = np.cos(theta_range[0]), np.cos(theta_range[1])
        theta = np.arccos(cos_min + v * (cos_max - cos_min))

        phi = phi_range[0] + w * (phi_range[1] - phi_range[0])

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.stack([x, y, z], axis=1)

    # ── 렌더링 ──────────────────────────────────────────────

    def _move_sensor(self, camera_pos: np.ndarray, target: np.ndarray = np.zeros(3)):
        """센서를 camera_pos로 이동시키고, target을 바라보게 한다."""
        q = self._look_at(camera_pos, target)
        self.sensor.set_local_pose(Pose(camera_pos, q))

    def get_extrinsic(self) -> np.ndarray:
        """현재 센서 자세 기준 4×4 extrinsic 행렬 반환."""
        T = self.sensor.get_pose().to_transformation_matrix()
        return self._R_S2O @ np.linalg.inv(T)

    def world_to_pixel(self, world_points: np.ndarray) -> np.ndarray:
        """
        월드 좌표 → 픽셀 좌표 변환.
        
        :param world_points: (3,) 단일 점 또는 (N, 3) 점 리스트
        :return: (2,) 단일 픽셀 또는 (N, 2) 픽셀 리스트
        """
        world_points = np.asarray(world_points, dtype=float)
        single = world_points.ndim == 1
        if single:
            world_points = world_points[np.newaxis, :]  # (1, 3)
 
        # (N, 3) → (N, 4) homogeneous
        ones = np.ones((len(world_points), 1))
        pts_h = np.hstack([world_points, ones])  # (N, 4)
 
        # projection: (4,4) @ (4,N) → (4,N)
        proj = (self.intrinsic @ self.get_extrinsic() @ pts_h.T).T  # (N, 4)
        z = proj[:, 2:3]
        z = np.where(np.abs(z) < 1e-6, np.nan, z)  # z≈0이면 nan 처리
        pixels = (proj[:, :2] / z)
 
        return pixels[0] if single else pixels
    
    def render(self, camera_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """
        지정 시점에서 depth 이미지를 렌더링한다.
        
        :param camera_pos: (3,) 카메라 월드 좌표
        :return: (H, W) float depth 배열
        """
        self._move_sensor(camera_pos,target_pos)
        self.scene.step()
        self.scene.update_render()
        self.sensor.take_picture()
        self.sensor.compute_depth()
        return self.sensor.get_depth()

    def render_spherical(
        self,
        num_views: int = 100,
        r_range: tuple = (0.2, 0.3),
        theta_range: tuple = (0, np.pi / 6),
        phi_range: tuple = (0, np.pi),
    ):
        """
        구면 좌표계 상에서 num_views만큼 시점을 샘플링하여 렌더링하는 제너레이터.
        
        Yields:
            (depth, extrinsic, camera_pos)
        """
        positions = self.sample_spherical_positions(
            r_range, theta_range, phi_range, num_views
        )
        for pos in positions:
            depth = self.render(pos)
            extrinsic = self.get_extrinsic()
            yield depth, extrinsic, pos

    # ── Grasp 이미지 크롭 ──────────────────────────────────

    @staticmethod
    def crop_grasp_image(
        image: np.ndarray,
        center: tuple,
        axis: tuple,
        crop_size: int = 256,
        output_size: int = 32,
    ) -> np.ndarray:
        """
        grasp 중심점 + 축 방향으로 이미지를 정렬·크롭·스케일링한다.
        
        :param image: (H,W) 또는 (H,W,C) 이미지
        :param center: 픽셀 좌표 (u, v)
        :param axis: 그립 방향 벡터 (du, dv)
        :param crop_size: 크롭 영역 한 변 크기
        :param output_size: 최종 출력 크기
        :return: (output_size, output_size) 크롭 이미지
        """
        angle = math.degrees(math.atan2(axis[1], axis[0]))
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        h, w = image.shape[:2]
        rotated = cv2.warpAffine(
            image, rot_mat, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        cropped = cv2.getRectSubPix(rotated, (crop_size, crop_size), center)
        return cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_AREA)

    # ── 시각화 유틸 ─────────────────────────────────────────

    @staticmethod
    def visualize_depth(depth: np.ndarray, point: tuple = None) -> np.ndarray:
        """depth 배열을 컬러맵 이미지로 변환. point가 주어지면 점을 찍는다."""
        norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        if point is not None:
            cv2.circle(colored, (int(point[0]), int(point[1])), 5, (0, 0, 0), -1)
        return colored

    @staticmethod
    def draw_grasp_debug(
        image: np.ndarray, center: tuple, axis: tuple, line_length: int = 200
    ) -> np.ndarray:
        """이미지에 grasp 중심점과 방향 축을 그려 반환한다."""
        vis = image.copy()
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(vis, (cx, cy), 10, (0, 0, 0), -1)

        mag = math.hypot(axis[0], axis[1])
        if mag > 0:
            ux, uy = axis[0] / mag, axis[1] / mag
            pt1 = (int(cx - ux * line_length), int(cy - uy * line_length))
            pt2 = (int(cx + ux * line_length), int(cy + uy * line_length))
            cv2.line(vis, pt1, pt2, (0, 0, 0), 4)
        return vis
 
# ── 사용 예시 ───────────────────────────────────────────────

if __name__ == "__main__":
    MESH = "/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/object/bin.stl"
    mesh=trimesh.load(MESH)
    trans,_=mesh.compute_stable_poses()
    renderer = GraspRenderer(MESH, mesh_scale=(0.001, 0.001, 0.001))

    # stable pose 예시 (단위 행렬 = 기본 자세)
    # 실제로는 mesh_stable_pose 등에서 계산된 회전 행렬을 넣으면 됨
    # renderer.set_stable_pose(stable_rotation_matrix)

    # 1) 단일 시점 렌더링
    while True:
        cam_pos = np.array([0.0, 0.0, 0.4])
        renderer.set_material(1.0,0.8)
        renderer.set_stable_pose(trans[1])
        depth = renderer.render(cam_pos)
        print(depth.shape)
        origin_px = renderer.world_to_pixel(np.array([0, 0, 0]))
        vis = renderer.visualize_depth(depth, point=origin_px)
        cv2.imshow("depth", vis)
        cv2.waitKey(0)

    # 2) 구면 좌표 다시점 렌더링 → DexNet 학습 데이터 생성
    for depth, extrinsic, pos in renderer.render_spherical(num_views=10):
        # DexNet에서 grasp candidate를 받았다고 가정
        # grasp_center_px, grasp_axis_px = dexnet.sample(...)
        # cropped = renderer.crop_grasp_image(depth, grasp_center_px, grasp_axis_px)
        pass