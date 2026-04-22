import sys,logging

import cv2,time
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from tf2_ros import Buffer, TransformListener, TransformBroadcaster,StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

from Minsoo_net.online.online_camera import RealSenseCamera
from Minsoo_net.online.online_sampler import OnlineAntipodalSampler,CrossEntropyRobustGraspingPolicy
from Minsoo_net.online.visualize import GraspVisualizer2D
from moveit_helper_functions import MoveItMoveHelper
sys.path.append('/home/minsoo/Dexnet_Minsoo/Minsoo_net/online')

class GraspPlannerNode(Node):
    def __init__(self, Checkpoint_path,use_visualize=False):
        super().__init__('Grasp_planning_node')
        self.viz=GraspVisualizer2D()
        self.camera=RealSenseCamera()
        
        self.depth=None
        self.image_size=None
        self.samples=None
        self.policy=CrossEntropyRobustGraspingPolicy(Checkpoint_path,OnlineAntipodalSampler,use_visualize=use_visualize)
        self.helper=MoveItMoveHelper()
        self.timer=self.create_timer(0.1,self.main_loop)
        self.timer=self.create_timer(0.1,self.tf_pub)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.visualize=use_visualize

    def main_loop(self):
        self.helper.move_to_joint_values(joint_goal={
            "joint_1": 0.2618,
            "joint_2": -0.0349,
            "joint_3": 1.8850,
            "joint_4": -0.0873,
            "joint_5": 1.0472,
            "joint_6": -1.2043
        })
        # self.helper.move_to_joint_values(joint_goal={
        #     "joint_1": 0.1945,
        #     "joint_2": 0.1722,
        #     "joint_3": 1.6341,
        #     "joint_4": 0.0021,
        #     "joint_5": 1.3097,
        #     "joint_6": -1.3113
        # })
        # self.helper.move_to_joint_values(joint_goal={
        #     "joint_1": 0.1920,
        #     "joint_2": 0.2094,
        #     "joint_3": 1.7279,
        #     "joint_4": 0.0000,
        #     "joint_5": 1.1868,
        #     "joint_6": -1.3090
        # })
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
        self.best_grasp,_=self.policy.cem_best(self.depth,num_iters=5)
        return self._pixel_to_world_coordinate(self.best_grasp,extrinsic)

    def _pixel_to_world_coordinate(self, grasp, extrinsic):
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

    def pick_and_place(self,pos,quat,offset_dir,offset):
        pos+=offset*offset_dir
        logging.debug(f' 다음 이동 Position: {pos}')
        input1=input('계속하려면 Enter')
        self.helper.move_cartesian(pos,quat)

        pos-=(offset+0.03)*offset_dir
        logging.debug(f' 다음 이동 Position: {pos}')
        input1=input('계속하려면 Enter')

        self.helper.move_cartesian(pos,quat)
        time.sleep(0.5)
        self.helper.gripper_close()

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
    logging.basicConfig(level=logging.DEBUG)
    rclpy.init()
    node = GraspPlannerNode('/home/minsoo/Dexnet_Minsoo/output/04-10_15-49_grasp_dataset_0408_CE_th0.002_a0.5_0.5/best.pt',use_visualize=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()