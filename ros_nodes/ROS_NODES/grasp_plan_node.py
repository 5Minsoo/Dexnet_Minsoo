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
        # filter=self._make_grasp_filter()
        self.best_grasp,_=self.policy.cem_best(depth_image=self.depth,num_iters=10, filter=None)
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
        # TODO: Sampling 단계에서 충돌 Grasp 제거 call back Filter 작성 필요. TF, Box -> Generates Collision grasp pre-remove filter. Input of filter -> (us,vs,theta,depth)
        p, r = self.get_tf('base_link', 'link_6')
        z_dir= -r.as_matrix()[2,2]
        p1, r1 = self.get_tf('link_6', 'hande_left_finger')
        p2, r2 = self.get_tf('link_6', 'hande_right_finger')
        finger=np.c_(p1.T,p2.T)
        def filter(grasps):
            N=len(grasps)
            uv1=np.stack([grasps[:,0], grasps[:,1], np.ones(N)])
            K_inv = np.linalg.inv(self.camera.intrinsic_parameter)
            cam=(K_inv @ uv1) * grasps[:,3]
            cam=np.vstack([cam, np.ones(N)])
            world = (self.get_extrinsic() @ cam)[:3].T

            theta=grasps[:,2]
            new_r=r * Rotation.from_euler('z', theta+np.pi/2)
            idx=(new_r*finger+world)[2,:]>self.config['box_z']
            return grasps[idx]
        return filter
        
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