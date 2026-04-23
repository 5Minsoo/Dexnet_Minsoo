import sys,logging

import cv2,time
import numpy as np
import argparse
import yaml
import math
from pathlib import Path
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, TransformBroadcaster,StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

from Minsoo_net.online.online_camera import RealSenseCamera
from Minsoo_net.online.online_sampler import OnlineAntipodalSampler,CrossEntropyRobustGraspingPolicy
from Minsoo_net.online.visualize import GraspVisualizer2D
from moveit_helper_functions import MoveItMoveHelper

sys.path.append('/home/minsoo/Dexnet_Minsoo/Minsoo_net/online')


class GraspPlannerNode(Node):
    def __init__(self, args, config):
        super().__init__('Grasp_planning_node')
        self.viz=GraspVisualizer2D()
        self.camera=RealSenseCamera()
        self.args=args
        self.config=config
        self.depth=None
        self.image_size=None
        self.samples=None
        self.visualize=self.args.visualize
        self.sampler=OnlineAntipodalSampler(gripper_width_m=self.config['gripper_width'], K=self.camera.intrinsic_parameter ,image_margin= self.config['image_margin'],max_edge=self.config['max_edge'],max_grasps=self.config['max_grasps'])
        self.policy=CrossEntropyRobustGraspingPolicy(self.config['model_path'],self.sampler,use_visualize=self.visualize)
        self.helper=MoveItMoveHelper()
        self.timer=self.create_timer(0.1,self.main_loop)
        self.timer=self.create_timer(0.1,self.tf_pub)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)


    def main_loop(self):
        self.helper.move_to_joint_values({k: math.radians(v) for k, v in self.config['tilts'][self.args.tilt].items()})
        self.helper.gripper_open()
        time.sleep(1.0)
        self.update_frame()
        pos,quat,offset_dir=self.plan_grasp(self.get_extrinsic())
        logging.debug(f' 물체 world (TCP 기준) Position: {pos}')
        if pos is not None:
            self.publish_grasp_tf(pos, quat)  # 추가
            self.pick_and_place(pos,quat,offset_dir,0.15)

                
    def update_frame(self):
        self.camera.update_frames()
        self.depth = self.camera.get_depth_image()

    def plan_grasp(self,extrinsic):
        filter=self._make_grasp_filter()
        self.best_grasp,_=self.policy.cem_best(self.depth,num_iters=10,filter=filter)
        if self.best_grasp is None:
            return None, None, None
        self.viz.visualize_from_grasps(self.depth._data,self.best_grasp,title="Best grasp")
        return self._pixel_to_world_coordinate(self.best_grasp,extrinsic)

    def _pixel_to_world_coordinate(self, grasp, extrinsic):
        if grasp is None:
            return None, None, None
        u, v, theta, z = grasp
        K = self.camera.intrinsic_parameter
        cam = np.linalg.inv(K) @ np.array([u, v, 1.0])
        cam *= z
        cam = np.append(cam, 1.0)
        logging.debug(f'카메라 좌표계 좌표: {cam}')

        # ── 물체 월드 좌표 (기존 유지) ──
        world = extrinsic @ cam
        obj_pos = world[:3].copy()
        logging.debug(f'물체의 월드 좌표계 좌표: {obj_pos}')

        # ── 현재 그리퍼 orientation ──
        t = self.tf_buffer.lookup_transform('base_link', 'link_6', rclpy.time.Time())
        r = t.transform.rotation
        R_grip = Rotation.from_quat([r.x, r.y, r.z, r.w])
        grip_z = -R_grip.as_matrix()[:3, 2]


        # ── 그리퍼 z축 기준으로 yaw만 회전 ──
        t = self.tf_buffer.lookup_transform('link_6', 'camera_link', rclpy.time.Time())
        r = t.transform.rotation
        dir_cam = np.array([np.cos(theta), np.sin(theta), 0])
        R_cam2grip = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
        dir_grip = R_cam2grip @ dir_cam
        yaw = np.arctan2(dir_grip[1], dir_grip[0]) + np.pi / 2

        yaw_rot = Rotation.from_euler('z', yaw)
        new_R = R_grip * yaw_rot
        quat = new_R.as_quat()
        logging.debug(f'물체 최종 월드 좌표: {obj_pos}, quat: {quat}')

        return obj_pos, quat,grip_z
        
    def get_extrinsic(self):
        t = self.tf_buffer.lookup_transform('base_link', 'camera_link', rclpy.time.Time())
        p = t.transform.translation
        r = t.transform.rotation
        mat = np.eye(4)
        mat[:3, :3] = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
        mat[:3, 3] = [p.x, p.y, p.z]
        return mat
    
    def publish_grasp_tf(self, pos, quat):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'grasp_pose'
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = pos
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = quat
        self.tf_broadcaster.sendTransform(t)
        self.last_tf=t
        
    def tf_pub(self):
        self.tf_broadcaster.sendTransform(self.last_tf)

    def _make_grasp_filter(self):
        """
        각 grasp을 실제로 실행(approach 축으로 z만큼 내려감)했을 때,
        더 낮은 finger tip이 박스 윗면(box_z)을 뚫지 않는 grasp만 통과.
        """
        t = self.tf_buffer.lookup_transform('base_link', 'link_6', rclpy.time.Time())
        p, r = t.transform.translation, t.transform.rotation
        R_grip = Rotation.from_quat([r.x, r.y, r.z, r.w])
        grip_origin = np.array([p.x, p.y, p.z])

        t_cg = self.tf_buffer.lookup_transform('link_6', 'camera_link', rclpy.time.Time())
        r_cg = t_cg.transform.rotation
        R_cam2grip = Rotation.from_quat([r_cg.x, r_cg.y, r_cg.z, r_cg.w])

        half_w = self.config['gripper_width'] / 2
        L = self.config['finger_length']
        tip_L0 = np.array([-half_w, 0, L])
        tip_R0 = np.array([ half_w, 0, L])

        box_z = self.config['box_z']
        max_penetration = self.config.get('max_penetration', 0.005)

        # 그리퍼 approach 방향 (월드 좌표계, 단위벡터)
        approach_dir = -R_grip.as_matrix()[:, 2]

        def _filter(samples):
            if len(samples) == 0:
                return samples

            theta_cam = samples[:, 2]
            z_target = samples[:, 3] - 0.118               # 각 grasp이 내려가는 거리
            N = len(theta_cam)

            # 카메라 yaw → 그리퍼 yaw
            dir_cam = np.stack([np.cos(theta_cam), np.sin(theta_cam), np.zeros(N)], axis=1)
            dir_grip = R_cam2grip.apply(dir_cam)
            yaw = np.arctan2(dir_grip[:, 1], dir_grip[:, 0]) + np.pi/2

            # Yaw 적용 후 finger tip의 월드 좌표 (현재 그리퍼 위치 기준)
            R_yaw = Rotation.from_euler('z', yaw)
            tip_L_world = R_grip.apply(R_yaw.apply(tip_L0)) + grip_origin
            tip_R_world = R_grip.apply(R_yaw.apply(tip_R0)) + grip_origin

            # 그리퍼가 approach 방향으로 z_target만큼 내려간 뒤의 finger tip 위치
            descent = z_target[:, None] * approach_dir   # (N, 3)
            tip_L_final = tip_L_world + descent
            tip_R_final = tip_R_world + descent

            # 더 낮은 finger가 박스 윗면 위에 있는지
            lowest_z = np.minimum(tip_L_final[:, 2], tip_R_final[:, 2])
            valid = lowest_z >= (box_z - max_penetration)
            return samples[valid]

        return _filter
        
    def pick_and_place(self,pos,quat,offset_dir,offset):
        pos+=offset*offset_dir
        logging.debug(f' 다음 이동 Position: {pos}')
        input1=input('계속하려면 Enter')
        self.helper.move_cartesian(pos,quat)

        pos-=(offset+self.config["hard_offset"])*offset_dir
        logging.debug(f' 다음 이동 Position: {pos}')
        input1=input('계속하려면 Enter')

        self.helper.move_cartesian(pos,quat)
        time.sleep(0.5)
        self.helper.gripper_close()
        time.sleep(0.5)
        pos+=offset*offset_dir
        self.helper.move_cartesian(pos,quat)
        time.sleep(0.5)
        pos-=offset*offset_dir
        self.helper.move_cartesian(pos,quat)
        time.sleep(0.5)
        self.helper.gripper_open()
        time.sleep(0.5)
        pos+=offset*offset_dir
        self.helper.move_cartesian(pos,quat)    
                
def main():    
    yaml_path=Path(__file__).parent.parent.parent.resolve() / "Minsoo_net" / "config" / "online_config.yaml"
    with open(yaml_path) as f:
        config=yaml.safe_load(f)

    parser = argparse.ArgumentParser(description="예제 스크립트")

    parser.add_argument("--tilt", "-t", default="vertical", help="Tilt 방향")
    parser.add_argument("--visualize", "-v", action='store_true', help="CEM 과정 시각화 스위치")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    rclpy.init()
    node = GraspPlannerNode(args,config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()