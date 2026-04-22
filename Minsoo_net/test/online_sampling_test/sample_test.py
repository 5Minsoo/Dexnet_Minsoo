import sys,logging,os

import cv2
import numpy as np
import torch

from Minsoo_net.online.online_camera import RealSenseCamera
from Minsoo_net.model.model import DexNet2
from Minsoo_net.online.online_sampler import OnlineAntipodalSampler, CrossEntropyRobustGraspingPolicy
from Minsoo_net.online.visualize import GraspVisualizer2D
from Minsoo_net.online.depth_image import DepthImage
sys.path.append('/home/minsoo/Dexnet_Minsoo/Minsoo_net/online')

class GraspPlanner:
    def __init__(self, Checkpoint_path, use_visualize=False, camera=False):
        self.viz=GraspVisualizer2D()
        if camera:
            self.camera=RealSenseCamera()
            self.K=self.camera.intrinsic_parameter
        else:
            self.camera=None
            self.K=None
        self.image=None
        self.depth=None
        self.samples=None
        self.visualize=use_visualize
        self.policy=CrossEntropyRobustGraspingPolicy(model_path=Checkpoint_path,sampler=OnlineAntipodalSampler,camera=False,use_visualize=use_visualize)

    def update_frame(self):
        self.camera.update_frames()
        self.depth = self.camera.get_depth_image()

    def plan_grasp(self):
        return self.policy.cem_best(self.depth,num_iters=10)

    def run(self,image=None):
        if image is None and self.K is not None:
            for _ in range(5):
                self.update_frame()
        else:
            self.depth=DepthImage(image)
            self.image=image
        logging.debug(f'self.depth._data shape{self.depth._data.shape}')
        logging.debug(f'입력 이미지 mean: {np.mean(a=self.depth._data)} std: {np.std(a=self.depth._data)}')
        grasp, score = self.plan_grasp()
        logging.debug(f'best grasp: {grasp}, score: {score}')
        return grasp, score

def main():
    logging.basicConfig(level=logging.DEBUG)
    planner = GraspPlanner('/home/minsoo/Dexnet_Minsoo/output/04-21_15-26_grasp_dataset_big1_CE_th0.002/epoch_017.pt',use_visualize=True,camera=False)
    # planner = GraspPlanner('/home/minsoo/Dexnet_Minsoo/output/Best_03-30/best.pt',use_visualize=True)
    for i in range(1,10):
        grasp, score = planner.run(
            image=cv2.imread(f'/home/minsoo/Dexnet_Minsoo/Minsoo_net/test/saved_data2/depth_raw_{i}.png', cv2.IMREAD_GRAYSCALE) * 0.001)
        if grasp is not None:
            logging.debug(f'Best grasp (u,v,theta,z): {grasp}, score: {score}')

if __name__ == '__main__':
    main()