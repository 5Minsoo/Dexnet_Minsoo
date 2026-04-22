import sys, logging, time, threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from scipy.spatial.transform import Rotation
from tf2_ros import Buffer, TransformListener

from Minsoo_net.online.online_camera import RealSenseCamera
from Minsoo_net.online.online_sampler import (
    OnlineAntipodalSampler, CrossEntropyRobustGraspingPolicy
)
from Minsoo_net.online.visualize import GraspVisualizer2D
from moveit_helper_functions import MoveItMoveHelper

sys.path.append('/home/minsoo/Dexnet_Minsoo/Minsoo_net/online')


HOME_JOINTS = {
    "joint_1": 0.2618,
    "joint_2": -0.0349,
    "joint_3": 1.8850,
    "joint_4": -0.0873,
    "joint_5": 1.0472,
    "joint_6": -1.2043,
}


class GraspPlannerNode(Node):
    def __init__(self, checkpoint_path, use_visualize=False):
        super().__init__('grasp_planning_node')
        
        # 컴포넌트 (한 번만 초기화)
        self.camera = RealSenseCamera()
        self.viz = GraspVisualizer2D()
        self.policy = CrossEntropyRobustGraspingPolicy(
            checkpoint_path, OnlineAntipodalSampler, use_visualize=use_visualize
        )
        self.helper = MoveItMoveHelper()
        
        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 최신 grasp 스냅샷: (pos, quat, offset_dir)
        self.latest_grasp = None
        
        # 로봇 이동 중에는 추론 건너뛰기 위한 플래그
        self.busy = False
        
        # 백그라운드 추론 타이머
        self.create_timer(0.3, self.inference_callback)
        
        self.get_logger().info('노드 초기화 완료')
    
    # ==========================================================
    # 백그라운드: 계속 돌면서 latest_grasp 갱신
    # ==========================================================
    def inference_callback(self):
        # 로봇 이동 중이면 depth가 가려질 수 있어서 스킵
        if self.busy:
            return
        
        try:
            self.camera.update_frames()
            depth = self.camera.get_depth_image()
            
            best_grasp, _ = self.policy.cem_best(depth, num_iters=5)
            self.viz.visualize_from_grasps(depth._data,best_grasp)
            extrinsic = self.get_extrinsic()
            pos, quat, offset_dir = self._pixel_to_world_coordinate(best_grasp, extrinsic)
            
            # 통째로 교체 (Lock 없이도 스냅샷 패턴으로 안전)
            self.latest_grasp = (pos, quat, offset_dir)
            self.get_logger().info(f'grasp 갱신: pos={pos}')
        
        except Exception as e:
            self.get_logger().warn(f'추론 실패: {e}')
    
    # ==========================================================
    # 키 입력 시 호출되는 액션들
    # ==========================================================
    def go_home(self):
        self.busy = True
        try:
            self.helper.move_to_joint_values(joint_goal=HOME_JOINTS)
            self.helper.gripper_open()
            time.sleep(1.0)
        finally:
            self.busy = False
    
    def execute_grasp(self):
        # 스냅샷 뜨기 (Lock 없이 안전)
        snapshot = self.latest_grasp
        if snapshot is None:
            print("아직 grasp 없음, 잠시 기다려주세요")
            return
        
        pos, quat, offset_dir = snapshot
        # pos를 복사해서 pick_and_place 내부에서 in-place 수정해도 원본 영향 없게
        self.busy = True
        try:
            self.pick_and_place(pos.copy(), quat.copy(), offset_dir.copy(), 0.15)
        finally:
            self.busy = False
    
    # ==========================================================
    # 기존 함수들 (그대로)
    # ==========================================================
    def _pixel_to_world_coordinate(self, grasp, extrinsic):
        u, v, theta, z = grasp
        K = self.camera.intrinsic_parameter
        cam = np.linalg.inv(K) @ np.array([u, v, 1.0])
        cam *= z
        cam = np.append(cam, 1.0)
        
        world = extrinsic @ cam
        obj_pos = world[:3].copy()
        
        t = self.tf_buffer.lookup_transform('base_link', 'link_6', rclpy.time.Time())
        r = t.transform.rotation
        R_grip = Rotation.from_quat([r.x, r.y, r.z, r.w])
        grip_z = -R_grip.as_matrix()[:3, 2]
        
        t = self.tf_buffer.lookup_transform('link_6', 'camera_link', rclpy.time.Time())
        r = t.transform.rotation
        dir_cam = np.array([np.cos(theta), np.sin(theta), 0])
        R_cam2grip = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
        dir_grip = R_cam2grip @ dir_cam
        yaw = np.arctan2(dir_grip[1], dir_grip[0]) + np.pi / 2
        
        yaw_rot = Rotation.from_euler('z', yaw)
        new_R = R_grip * yaw_rot
        quat = new_R.as_quat()
        
        return obj_pos, quat, grip_z
    
    def get_extrinsic(self):
        t = self.tf_buffer.lookup_transform('base_link', 'camera_link', rclpy.time.Time())
        p = t.transform.translation
        r = t.transform.rotation
        mat = np.eye(4)
        mat[:3, :3] = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
        mat[:3, 3] = [p.x, p.y, p.z]
        return mat
    
    def pick_and_place(self, pos, quat, offset_dir, offset):
        pos += offset * offset_dir
        self.helper.move_cartesian(pos, quat)
        
        pos -= (offset + 0.03) * offset_dir
        self.helper.move_cartesian(pos, quat)
        time.sleep(0.5)
        self.helper.gripper_close()
        
        pos += offset * offset_dir
        self.helper.move_cartesian(pos, quat)
        time.sleep(0.5)
        
        pos -= offset * offset_dir
        self.helper.move_cartesian(pos, quat)
        time.sleep(0.5)
        self.helper.gripper_open()
        time.sleep(0.5)
        
        pos += offset * offset_dir
        self.helper.move_cartesian(pos, quat)


# ==============================================================
# 메인: spin은 백그라운드, 키 입력은 메인 스레드
# ==============================================================
def main():
    logging.basicConfig(level=logging.INFO)
    rclpy.init()
    
    node = GraspPlannerNode(
        '/home/minsoo/Dexnet_Minsoo/output/04-10_15-49_grasp_dataset_0408_CE_th0.002_a0.5_0.5/best.pt',
        use_visualize=True,
    )
    
    # MultiThreadedExecutor로 spin을 백그라운드에서
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    # 시작 시 홈으로
    time.sleep(1.0)
    node.go_home()
    
    print("=" * 50)
    print("  [g] 현재 타겟으로 pick & place")
    print("  [h] 홈 위치로 복귀")
    print("  [q] 종료")
    print("=" * 50)
    
    try:
        while rclpy.ok():
            key = input("> ").strip().lower()
            
            if key == 'g':
                node.execute_grasp()
                node.go_home()  # pick&place 후 자동으로 홈 복귀
            elif key == 'h':
                node.go_home()
            elif key == 'q':
                break
            else:
                print("g / h / q 중 하나 입력")
    
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()