from unittest import TestCase
import logging

import zarr
import numpy as np

from Minsoo_net.grasp.collision_checker import GraspCollisionChecker
from Minsoo_net.grasp.gripper import RobotGripper
from Minsoo_net.grasp.Object import GraspableObject3D
from Minsoo_net.grasp.grasp_sampler import AntipodalGraspSampler

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
zarr_path='/home/minsoo/Dexnet_Minsoo/grasp_dataset_260000.zarr'

class TestDataset(TestCase):
    def setUp(self):
        self.f=zarr.open(zarr_path,mode='r')

    def tearDown(self):
        None

    def test_dataset_length(self):
        keys=list(self.f.keys())
        for i, key in enumerate(keys):
            poses=list(self.f[key].keys())
            for i,pose in enumerate(poses):
                len_image=self.f[key][pose]['images'].shape[0]
                len_label=self.f[key][pose]['labels'].shape[0]
                len_gripper=self.f[key][pose]['gripper_depth'].shape[0]
                self.assertEqual(len_image,len_label)
                self.assertEqual(len_label,len_gripper)
    
    def test_dataset_dimension(self):
        keys=list(self.f.keys())
        for i, key in enumerate(keys):
            poses=list(self.f[key].keys())
            for i,pose in enumerate(poses):
                shape=self.f[key][pose]['images'].shape
                self.assertEqual(len(shape),3)
                self.assertEqual(shape[1],32)
                self.assertEqual(shape[2],32)
    
    def test_dataset_clean(self):
        keys = list(self.f.keys())
        for i, key in enumerate(keys):
            poses = list(self.f[key].keys())
            for j, pose in enumerate(poses):
                img = self.f[key][pose]['images'][:]
                self.assertFalse(np.any(np.isnan(img)))