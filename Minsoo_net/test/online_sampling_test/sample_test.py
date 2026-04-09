import sys,logging,os

import cv2
import numpy as np
import torch

from Minsoo_net.online.online_camera import RealSenseCamera
from Minsoo_net.model.model import DexNet2
from Minsoo_net.online.online_sampler import OnlineAntipodalSampler
from Minsoo_net.online.visualize import GraspVisualizer2D
from Minsoo_net.online.depth_image import DepthImage
sys.path.append('/home/minsoo/Dexnet_Minsoo/Minsoo_net/online')

class GraspPlanner:
    def __init__(self, Checkpoint_path, use_visualize=False, camera=False):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.viz=GraspVisualizer2D()
        self.model=DexNet2.load(path=Checkpoint_path)
        self.model.to(device)
        if camera:
            self.camera=RealSenseCamera()
            self.K=self.camera.intrinsic_parameter
        else:
            self.camera=None
            self.K=None
        self.image=None
        self.depth=None
        self.sampler=OnlineAntipodalSampler(gripper_width_m=0.05,K=self.K,image_margin=0.2,max_edge=100)
        self.samples=None
        self.visualize=use_visualize

    def update_frame(self):
        self.camera.update_frames()
        self.depth = self.camera.get_depth_image()

    def collect_samples(self):
        samples=self.sampler.sample_grasps(self.depth,use_visualize=self.visualize)
        self.samples=samples

    def plan_grasp(self):
        all_samples = self.samples
        cropped = RealSenseCamera.crop_and_rotate_batch(
            self.depth._data, all_samples, crop_size=(96, 96))
        cropped = np.array([
            cv2.resize(self.depth._data, (32, 32), interpolation=cv2.INTER_CUBIC)
            for self.depth._data in cropped
        ])
        
        cropped_input = np.expand_dims(cropped, axis=-1)
        poses_input = all_samples[:, 3].reshape(-1, 1)

        success_probs = self.model.predict_success(cropped_input, poses_input)

        best_idx = np.argmax(success_probs)
        self.best_grasp = all_samples[best_idx]
        self.best_score = success_probs[best_idx]
        cv2.imshow(f'cropped', cropped[best_idx])
        cv2.waitKey(0)
        if self.visualize:
            self.viz.visualize_debug(self.image,all_samples,success_probs)

        save_dir = '/home/minsoo/Dexnet_Minsoo/Minsoo_net/test/saved_data2'
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, 'grasp_data.npz'),
                 depth=self.depth._data,
                 cropped=cropped_input,
                 poses=poses_input,
                 samples=all_samples,
                 success_probs=success_probs)
        # depth_uint16 = self.depth._data.astype(np.uint16)
        cv2.imwrite(os.path.join(save_dir, 'depth_raw.png'), self.depth._data*1000)
        logging.debug(f'saved to {save_dir}')

        return self.best_grasp, self.best_score

    def run(self,image=None):
        if image is None and self.K is not None:
            for _ in range(5):
                self.update_frame()
        else:
            self.depth=DepthImage(image)
            self.image=image
        logging.debug(f'self.depth._data shape{self.depth._data.shape}')
        logging.debug(f'입력 이미지 mean: {np.mean(a=self.depth._data)} std: {np.std(a=self.depth._data)}')
        self.collect_samples()
        if len(self.samples)>0:
            grasp, score = self.plan_grasp()
            logging.debug(f'best grasp: {grasp}, score: {score}')
            return grasp, score
        return None, None

def main():
    logging.basicConfig(level=logging.DEBUG)
    planner = GraspPlanner('/home/minsoo/Dexnet_Minsoo/output/04-08_grasp_dataset_0408_CE_th0.002_a0.5_0.5/best.pt',use_visualize=True,camera=False)
    # planner = GraspPlanner('/home/minsoo/Dexnet_Minsoo/output/Best_03-30/best.pt',use_visualize=True)
    grasp, score = planner.run(image=cv2.imread('/home/minsoo/Dexnet_Minsoo/Minsoo_net/test/saved_data/depth_raw_1.png',cv2.IMREAD_GRAYSCALE)*0.001)
    if grasp is not None:
        logging.debug(f'Best grasp (u,v,theta,z): {grasp}, score: {score}')

if __name__ == '__main__':
    main()