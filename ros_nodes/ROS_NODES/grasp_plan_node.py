import sys,logging

import cv2,time
cv2.setNumThreads(0)
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
from rclpy.logging import LoggingSeverity

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
        self.sampler=OnlineAntipodalSampler(gripper_width_m=self.config['gripper_width'], K=self.camera.intrinsic_parameter ,image_margin= self.config['image_margin'],max_edge=self.config['max_edge'],max_grasps=self.config['max_grasps'],visualize=self.visualize)
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
        self.best_grasp,_=self.policy.cem_best(depth_image=self.depth,num_iters=10, filter=filter)
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
        p, R_grip= self.get_tf('base_link', 'link_6')
        grip_z = -R_grip.as_matrix()[:3, 2]

        # ── 그리퍼 z축 기준으로 yaw만 회전 ──
        p, r= self.get_tf('link_6', 'camera_link')
        dir_cam = np.array([np.cos(theta), np.sin(theta), 0])
        R_cam2grip = r.as_matrix()
        dir_grip = R_cam2grip @ dir_cam
        yaw = np.arctan2(dir_grip[1], dir_grip[0]) + np.pi / 2

        yaw_rot = Rotation.from_euler('z', yaw)
        new_R = R_grip * yaw_rot
        quat = new_R.as_quat()
        logging.debug(f'물체 최종 월드 좌표: {obj_pos}, quat: {quat}')

        return obj_pos, quat,grip_z
        
    def get_extrinsic(self):
        p,r = self.get_tf('base_link','camera_link')
        mat = np.eye(4)
        mat[:3, :3] = r.as_matrix()
        mat[:3, 3] = p
        return mat
    
    def get_tf(self, start, end):
        t = self.tf_buffer.lookup_transform(start, end, rclpy.time.Time())
        
        p = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
        q = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
        
        r = Rotation.from_quat(q) 
        
        return p, r
    
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
        _, R_grip     = self.get_tf('base_link', 'hande_tcp_link')   # 현재 그리퍼 자세
        _, R_cam2tcp = self.get_tf('hande_tcp_link', 'camera_link')
        R_cam2tcp    = R_cam2tcp.as_matrix()
        p_left,  _    = self.get_tf('hande_tcp_link', 'hande_left_finger')   # link_6 기준 손가락 끝 위치
        p_right, _    = self.get_tf('hande_tcp_link', 'hande_right_finger')
        
        K_inv     = np.linalg.inv(self.camera.intrinsic_parameter)
        extrinsic = self.get_extrinsic()
        box_z     = self.config['box_z']
        margin    = 0.0   # 필요하면 안전 여유(m) 추가

        def grasp_filter(grasps):
            grasps = np.atleast_2d(np.asarray(grasps, dtype=float))
            N = grasps.shape[0]
            if N == 0:
                return grasps
            u, v, theta, z = grasps[:, 0], grasps[:, 1], grasps[:, 2], grasps[:, 3]

            # 1) pixel -> camera -> world (grasp 중심 = grasp 시 link_6 위치)
            uv1   = np.stack([u, v, np.ones(N)])      # (3, N)
            cam   = (K_inv @ uv1) * z                  # (3, N), 각 열을 z로 스케일
            cam   = np.vstack([cam, np.ones(N)])       # (4, N)
            world = (extrinsic @ cam)[:3].T            # (N, 3)

            # 2) 이미지 theta -> 그리퍼 yaw 
            dir_cam  = np.stack([np.cos(theta), np.sin(theta), np.zeros(N)])  # (3, N)
            dir_tcp  = R_cam2tcp @ dir_cam                                     # (3, N)
            yaw      = np.arctan2(dir_tcp[1], dir_tcp[0]) + np.pi / 2 

            # 3) grasp 자세 = R_grip * Rz(yaw)  (단일 회전 * 스택 브로드캐스트)
            #    scipy 1.17 에서는 1D 각도가 안 먹혀 (-1,1) 로 reshape 필요
            R_grasp = R_grip * Rotation.from_euler('z', yaw.reshape(-1, 1))

            # 4) 손가락 양끝 월드 z = world_z + (R_grasp @ p_finger)_z
            tip_left_z  = world[:, 2] + R_grasp.apply(p_left)[:, 2]
            tip_right_z = world[:, 2] + R_grasp.apply(p_right)[:, 2]

            # 5) 더 낮은 손가락 끝이 바닥 아래면 제거
            lowest = np.minimum(tip_left_z, tip_right_z)
            keep   = lowest > (box_z + margin)
            keep_grasps = grasps[keep]
            print(f'Filtered { N - keep_grasps.shape[0] }')
            return keep_grasps

        return grasp_filter
        
    def pick_and_place(self,pos,quat,offset_dir,offset):
        pos1=pos+offset*offset_dir
        i = input(f'다음 이동 Position: {pos1} 이동하려면 Enter 취소: q  ')
        if i == 'q':
            return
        self.helper.move_cartesian(pos1,quat)

        time.sleep(0.3)
        pos2=pos+0.05*offset_dir
        self.helper.move_cartesian(pos2,quat)

        pos3=pos+self.config["hard_offset"]*offset_dir
        i = input(f'다음 이동 Position: {pos3} 이동하려면 Enter 취소: q  ')
        if i == 'q':
            return
        
        self.helper.move_cartesian(pos3,quat)
        time.sleep(0.5)
        self.helper.gripper_close()
        time.sleep(0.5)
        pos4=pos+0.15*offset_dir
        self.helper.move_cartesian(pos4,quat)
        time.sleep(0.5)
        self.helper.move_cartesian(pos,quat)
        time.sleep(0.5)
        self.helper.gripper_open()
        # place = np.array(self.config['place'])
        # place1 = place.copy()
        # place1[2]+= 0.20
        # self.helper.move_cartesian(place1,quat)
        # time.sleep(0.5)
        # place = self.config['place']
        # self.helper.move_cartesian(place,quat) 
        # self.helper.gripper_open()
        # time.sleep(0.5)
        # self.helper.move_cartesian(place1,quat)
        # time.sleep(0.5)
        # place1[2] += 0.20
        # self.helper.move_cartesian(place1,quat)    
                
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