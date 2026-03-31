import sys,logging,os

import cv2
import numpy as np
import torch

from Minsoo_net.online.online_camera import RealSenseCamera
from Minsoo_net.model.model import DexNet2
from Minsoo_net.online.online_sampler import OnlineAntipodalSampler
from Minsoo_net.online.visualize import GraspVisualizer2D
sys.path.append('/home/minsoo/Dexnet_Minsoo/Minsoo_net/online')

class GraspPlanner:
    def __init__(self, Checkpoint_path, use_visualize=False):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.viz=GraspVisualizer2D()
        self.model=DexNet2.load(path=Checkpoint_path)
        self.model.to(device)
        self.camera=RealSenseCamera()

        self.depth=None
        self.sampler=OnlineAntipodalSampler(gripper_width_m=0.05,K=self.camera.intrinsic_parameter)
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
            cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
            for img in cropped
        ])
        for i, image in enumerate(cropped):
            cv2.imshow(f'cropped {i}', image)
            cv2.waitKey(0)
        
        cropped_input = np.expand_dims(cropped, axis=-1)
        poses_input = all_samples[:, 3].reshape(-1, 1)

        success_probs = self.model.predict_success(cropped_input, poses_input)

        best_idx = np.argmax(success_probs)
        self.best_grasp = all_samples[best_idx]
        self.best_score = success_probs[best_idx]
        if self.visualize:
            self.viz.visualize_debug(self.depth._data,all_samples,success_probs)

        save_dir = '/home/minsoo/Dexnet_Minsoo/Minsoo_net/test/saved_data1'
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, 'grasp_data.npz'),
                 depth=self.depth._data,
                 cropped=cropped_input,
                 poses=poses_input,
                 samples=all_samples,
                 success_probs=success_probs)
        depth_uint16 = self.depth._data.astype(np.uint16)
        cv2.imwrite(os.path.join(save_dir, 'depth_raw.png'), depth_uint16)
        logging.debug(f'saved to {save_dir}')

        return self.best_grasp, self.best_score

    def run(self):
        self.update_frame()
        self.collect_samples()
        if len(self.samples)>0:
            grasp, score = self.plan_grasp()
            logging.debug(f'best grasp: {grasp}, score: {score}')
            return grasp, score
        return None, None

def main():
    logging.basicConfig(level=logging.DEBUG)
    planner = GraspPlanner('/home/minsoo/Dexnet_Minsoo/output/Best_03-30/best.pt',use_visualize=True)
    grasp, score = planner.run()
    if grasp is not None:
        logging.debug(f'Best grasp (u,v,theta,z): {grasp}, score: {score}')

if __name__ == '__main__':
    main()