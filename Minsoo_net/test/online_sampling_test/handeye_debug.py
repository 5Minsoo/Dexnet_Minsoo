import sys
import os
import time
import math
import numpy as np
import yaml
import cv2

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation

# MoveIt 헬퍼 및 RealSense 카메라 클래스 임포트
current_dir = os.path.dirname(os.path.abspath(__file__))
helper_path = os.path.abspath(os.path.join(current_dir, '../../../bin_picking'))
sys.path.append(helper_path)

from ROS_NODES.ROS_NODES.moveit_helper_functions import MoveItMoveHelper
from Minsoo_net.online.online_camera import RealSenseCamera

class HandEyeValidator(MoveItMoveHelper):
    def __init__(self):
        super().__init__() 
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.get_logger().info("Initializing RealSense Camera...")
        self.camera = RealSenseCamera()
        
        # ArUco 마커 설정 (실제 길이에 맞게 0.05m = 5cm 세팅 확인)
        self.marker_length = 0.2
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        l = self.marker_length / 2.0
        self.obj_points = np.array([
            [-l,  l, 0],
            [ l,  l, 0],
            [ l, -l, 0],
            [-l, -l, 0]
        ], dtype=np.float32)

        self.marker_positions = []
        # === [추가] 뎁스 에러를 모아둘 리스트 ===
        self.depth_errors = []

    def run_validation(self, target_xyz, radius=0.45, tilt_deg=30, num_shots=8):
        self.get_logger().info("=== Starting Integrated Hand-Eye & Depth Validation ===")
        target = np.array(target_xyz)
        tilt_rad = math.radians(tilt_deg)
        pan_angles = np.linspace(0, 360, num_shots, endpoint=False)

        for i, pan in enumerate(pan_angles):
            pan_rad = math.radians(pan)

            cam_pos = [
                target[0] + radius * math.sin(tilt_rad) * math.cos(pan_rad),
                target[1] + radius * math.sin(tilt_rad) * math.sin(pan_rad),
                target[2] + radius * math.cos(tilt_rad)
            ]
            direction = target - np.array(cam_pos)
            q_obj = self.make_grasp_quat_for_approach(-direction, np.array([1, 0, 0]))

            try:
                t_cam = self.tf_buffer.lookup_transform('link_6', 'camera_link', rclpy.time.Time())
                camera_offset = np.array([
                    t_cam.transform.translation.x, t_cam.transform.translation.y, t_cam.transform.translation.z
                ])
            except Exception as e:
                camera_offset = np.array([-0.08, -0.01, 0.04]) 

            rot = Rotation.from_quat([q_obj.x, q_obj.y, q_obj.z, q_obj.w])
            gripper_pos = np.array(cam_pos) - rot.apply(camera_offset)

            if self.move_cartesian(gripper_pos.tolist(), [q_obj.x, q_obj.y, q_obj.z, q_obj.w]):
                self._wait_with_spin(1.0) 
                
                for _ in range(3):
                    self.camera.update_frames()
                
                color_img = self.camera.get_color_image()
                # === [추가] 뎁스 이미지도 같이 가져오기 ===
                depth_img_obj = self.camera.get_depth_image() 
                
                K = self.camera.intrinsic_parameter
                dist_coeffs = np.zeros((5, 1)) 

                corners, ids, _ = self.detector.detectMarkers(color_img)
                
                if ids is not None and depth_img_obj is not None:
                    success, rvec, tvec = cv2.solvePnP(self.obj_points, corners[0][0], K, dist_coeffs)
                    
                    if success:
                        cam_point = tvec.flatten()
                        aruco_z = cam_point[2] # 알고리즘이 푼 마커의 Z 거리
                        
                        # === [추가] 뎁스 센서 값과 ArUco Z값 비교 로직 ===
                        # 1. 픽셀 좌표(u, v) 평균을 내서 마커의 정중앙 좌표 구하기
                        c_u = int(np.mean(corners[0][0][:, 0]))
                        c_v = int(np.mean(corners[0][0][:, 1]))
                        
                        # 2. 넘파이 배열에서 해당 픽셀의 깊이 읽어오기 (미터 단위 가정)
                        depth_array = depth_img_obj._data 
                        sensor_z = float(depth_array[c_v, c_u])
                        
                        # 3. Z값이 0(측정 실패)이 아닐 때만 에러 계산
                        if sensor_z > 0.0:
                            depth_err = abs(aruco_z - sensor_z)
                            self.depth_errors.append(depth_err)
                            self.get_logger().info(f"[{i+1}/{num_shots}] Depth Check -> ArUco: {aruco_z*1000:.1f}mm | Sensor: {sensor_z*1000:.1f}mm | Error: {depth_err*1000:.1f}mm")
                        else:
                            self.get_logger().warn(f"[{i+1}/{num_shots}] Sensor Z is 0.0 (Invalid depth at marker center)")
                        # ==================================================

                        try:
                            t_base_cam = self.tf_buffer.lookup_transform('base_link', 'camera_link', rclpy.time.Time())
                            tf_trans = np.array([t_base_cam.transform.translation.x, 
                                                 t_base_cam.transform.translation.y, 
                                                 t_base_cam.transform.translation.z])
                            tf_rot = Rotation.from_quat([t_base_cam.transform.rotation.x,
                                                         t_base_cam.transform.rotation.y,
                                                         t_base_cam.transform.rotation.z,
                                                         t_base_cam.transform.rotation.w])
                            
                            marker_base = tf_rot.apply(cam_point) + tf_trans
                            self.marker_positions.append(marker_base)
                            
                        except Exception as e:
                            self.get_logger().error(f"TF Transform failed: {e}")
                else:
                    self.get_logger().warn(f"[{i+1}/{num_shots}] ArUco marker NOT detected or Depth missing!")

            else:
                self.get_logger().warn(f"Move failed for pan={pan:.1f} — skipping")

        self._calculate_and_print_error()

    def _calculate_and_print_error(self):
        if len(self.marker_positions) < 2:
            self.get_logger().warn("Not enough data to calculate calibration error.")
            return

        pts = np.array(self.marker_positions)
        mean_pt = np.mean(pts, axis=0)
        std_dev = np.std(pts, axis=0)
        max_diff = np.max(pts, axis=0) - np.min(pts, axis=0)
        
        # 뎁스 센서 오차 평균 계산
        mean_depth_err = np.mean(self.depth_errors) if self.depth_errors else 0.0
        
        self.get_logger().info("\n" + "="*50)
        self.get_logger().info(" 📊 Hand-Eye & Depth Calibration Error Report ")
        self.get_logger().info("="*50)
        self.get_logger().info(f"Shots successfully detected: {len(pts)}")
        self.get_logger().info(f"Center (Mean) : [{mean_pt[0]:.4f}, {mean_pt[1]:.4f}, {mean_pt[2]:.4f}] m")
        self.get_logger().info(f"Max Diff      : [{max_diff[0]*1000:.2f}, {max_diff[1]*1000:.2f}, {max_diff[2]*1000:.2f}] mm")
        self.get_logger().info(f"Std Dev (RMS) : [{std_dev[0]*1000:.2f}, {std_dev[1]*1000:.2f}, {std_dev[2]*1000:.2f}] mm")
        self.get_logger().info("-" * 50)
        self.get_logger().info(f"🔍 Sensor Z vs ArUco Z Avg Error: {mean_depth_err*1000:.2f} mm")
        self.get_logger().info("="*50)

    def _wait_with_spin(self, duration):
        start = time.time()
        while time.time() - start < duration:
            rclpy.spin_once(self, timeout_sec=0.1)

def main():
    rclpy.init()
    validator = HandEyeValidator()
    
    # 목표 좌표 주변을 돌며 검증 수행
    validator.run_validation(target_xyz=[-0.699, -0.067, 0.1], radius=0.45, tilt_deg=30, num_shots=8)
    
    validator.camera.release()
    validator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()