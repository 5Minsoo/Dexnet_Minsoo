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
        self.f=zarr.open(zarr_path,'r',zarr_version=3)

    def tearDown(self):
        None

    def test_dataset_length(self):
        print(self.f.keys())