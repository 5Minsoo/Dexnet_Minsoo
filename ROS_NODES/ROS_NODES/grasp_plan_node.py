from Minsoo_net.online.online_camera import RealSenseCamera
from Minsoo_net.model.model import DexNet2
from Minsoo_net.online.online_sampler import OnlineAntipodalSampler
import rclpy
from rclpy.node import Node
import torch
import numpy as np
import cv2,time
from scipy.spatial.transform import Rotation
from moveit_helper_functions import MoveItMoveHelper
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class GraspPlannerNode(Node):
    def __init__(self, Checkpoint_path):
        super().__init__('Grasp_planning_node')
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model=DexNet2.load(path=Checkpoint_path)
        self.model.to(device)
        self.camera=RealSenseCamera()
        
        self.color=None
        self.depth=None
        self.image_size=None

        self.sampler=OnlineAntipodalSampler(0.05,self.camera.intrinsic_parameter)
        self.sample_buffer=[]
        self.sampling_time=3.0
        self.sampling_start=time.time()

        self.helper=MoveItMoveHelper()
        self.timer=self.create_timer(0.1,self.main_loop)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.roi = None

    def main_loop(self):
        self.helper.move_to_joint_values(joint_goal={
            "joint_1": 0.2269,
            "joint_2": 0.1745,
            "joint_3": 1.6232,
            "joint_4": 0.0,
            "joint_5": 1.3439,
            "joint_6": -1.3439
        })
        if self.roi is None:
            self.roi = self._select_roi()
            return
        
        self.update_frame()
        self.collect_samples()

        if time.time()-self.sampling_start>=self.sampling_time:
            if len(self.sample_buffer) == 0:
                self.sampling_start = time.time()
                return
            pos,quat=self.plan_grasp(self.get_extrinsic())
            self.sample_buffer.clear()
            self.sampling_start=time.time()
        
            if pos is not None:
                self.publish_grasp_tf(pos, quat)  # 추가
                self.helper.move_pick_and_place(pos,quat,0.15)
        
        cv2.imshow("ROI", self.color)
        cv2.waitKey(0)

        
    def update_frame(self):
        self.camera.update_frames()
        color, depth = self.camera.inside_box_image()
        self.depth = self.apply_roi_mask(depth) * self.camera.depth_scale
        self.color = self.apply_roi_mask(color)
        self.image_size = depth.shape[1]

        samples = self.sampler.sample_grasps(self.depth)
        if samples is not None and len(samples) > 0:
            self.sample_buffer.append(samples)

    def collect_samples(self):
        samples=self.sampler.sample_grasps(self.depth)
        if len(samples)>0:
            self.sample_buffer.append(samples)

    def plan_grasp(self,extrinsic):
        all_samples = np.vstack(self.sample_buffer)
        all_samples = np.unique(all_samples, axis=0)
        cropped = RealSenseCamera.crop_and_rotate_batch(
        self.depth, all_samples, crop_size=(96, 96))
        cropped = np.array([
            cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
            for img in cropped
        ])
        cropped_input = np.expand_dims(cropped, axis=-1)
        poses_input = all_samples[:, 3].reshape(-1, 1)

        success_probs = self.model.predict_success(cropped_input, poses_input)

        best_idx = np.argmax(success_probs)
        self.best_grasp = all_samples[best_idx]
        self.best_score = success_probs[best_idx]

        return self._pixel_to_world_coordinate(self.best_grasp,extrinsic=extrinsic)

    def _pixel_to_world_coordinate(self, grasp, extrinsic):
        u, v, theta, z = grasp
        K = self.camera.intrinsic_parameter
        T = np.eye(4)
        T[:3, :3] = K

        # ── 위치 변환 ──
        cam = np.linalg.inv(T) @ np.array([u, v, 1, 1])
        cam[:3] *= z
        world = extrinsic @ cam

        # ── theta 변환 ──
        R = extrinsic[:3, :3]
        dir_cam = np.array([np.cos(theta), np.sin(theta), 0])
        dir_world = R @ dir_cam
        yaw = np.arctan2(dir_world[1], dir_world[0])

        rpy = Rotation.from_matrix(R).as_euler('xyz')
        roll, pitch = rpy[0], rpy[1]
        quat = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_quat()

        return world[:3], quat
    
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

    def _select_roi(self):
        self.camera.update_frames()
        color=self.camera.get_color_image() 
        depth = self.camera.get_depth_image()
        cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
        roi = cv2.selectROI("ROI", depth)
        # 확인용
        x, y, w, h = roi
        masked = np.zeros_like(color)
        masked[y:y+h, x:x+w] = color[y:y+h, x:x+w]
        cv2.imshow("ROI", masked)
        cv2.waitKey(1)
        return roi

    def apply_roi_mask(self, image):
        mask = np.zeros_like(image)
        x, y, w, h = self.roi
        mask[y:y+h, x:x+w] = image[y:y+h, x:x+w]
        return mask
    

def main():
    rclpy.init()
    node = GraspPlannerNode('/home/minsoo/Dexnet_Minsoo/output/20260324_15-07/best.pt')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()