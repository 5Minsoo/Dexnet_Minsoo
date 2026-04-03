from unittest import TestCase
import logging

import zarr
import numpy as np

from Minsoo_net.grasp.collision_checker import GraspCollisionChecker
from Minsoo_net.grasp.gripper import RobotGripper
from Minsoo_net.grasp.Object import GraspableObject3D
from Minsoo_net.online.online_sampler import OnlineAntipodalSampler
from Minsoo_net.model.model import DexNet2
from Minsoo_net.test.online_sampling_test.sample_test import GraspPlanner
from Minsoo_net.online.online_camera import RealSenseCamera
from Minsoo_net.online.depth_image import DepthImage
import cv2
import torch
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
config_path='/home/minsoo/Dexnet_Minsoo/Minsoo_net/config/master_config.yaml'
zarr_path='/home/minsoo/Dexnet_Minsoo/grasp_dataset.zarr'
model_path='/home/minsoo/Dexnet_Minsoo/output/Best_03-30/best.pt'

class TestDataset(TestCase):
    def setUp(self):
        self.model_fresh=DexNet2()
        self.model_new=DexNet2.load(model_path)
        self.sampler=OnlineAntipodalSampler(gripper_width_m=0.05)

    def tearDown(self):
        None

    def test_model_weight(self):
        for key in self.model_fresh.state_dict().keys():
            self.assertFalse(torch.allclose(self.model_fresh.state_dict()[key],self.model_new.state_dict()[key]))

    def test_model_validity(self):
        depth_data=cv2.imread('/home/minsoo/Dexnet_Minsoo/Minsoo_net/test/saved_data/depth_raw_1.png',-1)*0.001
        dummy_depth = DepthImage(depth_data)
        samples=self.sampler.sample_grasps(depth_image=dummy_depth,use_visualize=False)
        self.samples=samples
        all_samples = self.samples
        cropped = RealSenseCamera.crop_and_rotate_batch(
            dummy_depth._data, all_samples, crop_size=(96, 96))
        cropped = np.array([
            cv2.resize(dummy_depth._data, (32, 32), interpolation=cv2.INTER_CUBIC)
            for dummy_depth._data in cropped
        ])
        
        cropped_input = np.expand_dims(cropped, axis=-1)
        poses_input = all_samples[:, 3].reshape(-1, 1)

        success_probs = torch.tensor(self.model_new.predict(cropped_input, poses_input))
        self.assertFalse(torch.all(success_probs[:,0] > 0.9), "모델이 1.0에 너무 치우쳐 있습니다.")
        self.assertFalse(torch.all(success_probs[:,1] < 0.0), "모델이 1.0에 너무 치우쳐 있습니다.")