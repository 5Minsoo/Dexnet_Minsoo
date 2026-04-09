from unittest import TestCase
import logging

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
yaml_path = '/home/minsoo/Dexnet_Minsoo/Minsoo_net/config/master_config.yaml'
obj_path = '/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/object/bin.stl'

gripper = RobotGripper('hande', yaml_path)
checker = GraspCollisionChecker(gripper,use_visual=True)
sampler = AntipodalGraspSampler(yaml_path)
obj = GraspableObject3D(obj_path)
stable_poses, probs = obj.stable_poses()
pose=stable_poses[0]
grasps = sampler.sample_grasps(obj, 10)
aligned_grasps = [grasp.perpendicular_table(pose) for grasp in grasps]
vertical_grasp=[]

for grasp in aligned_grasps:
    _, approach_angle, _ = grasp.grasp_angles_from_stp_z(pose)
    
    if approach_angle < np.deg2rad(15):
        vertical_grasp.append(grasp)

class TestCollisionProb(TestCase):
    def setUp(self):
        self.key = 'bin'
        checker.set_table()
        checker.set_object(self.key, obj.mesh, pose)

    def tearDown(self):
        checker.remove_object(self.key)

    def test_collision_prob_debug(self):
        print()
        for i, grasp in enumerate(vertical_grasp,1):
            prob = checker.collision_prob(obj, grasp, key=self.key, num=5)
            print(f'{i}번째 grasp 충돌확률: {prob}')

    def test_object_approach_angle_in_range(self):
        print()
        for i, grasp in enumerate(aligned_grasps):
            _, approach_angle, _ = grasp.grasp_angles_from_stp_z(pose)
            print(f"grasp[{i}]: approach_angle={np.degrees(approach_angle):.2f}°")
            assert -np.pi/2 <= approach_angle <= np.pi/2
            

    # def test_transforms_restored_after_collision_prob(self):
    #     tf_before = checker._objs_tf[self.key].copy()
    #     checker.collision_prob(obj, grasps[0], key=self.key, num=10)
    #     tf_after = checker._objs_tf[self.key]
    #     np.testing.assert_array_almost_equal(tf_before, tf_after)

    # def test_collision_prob_multiple_grasps(self):
    #     probs = []
    #     for g in grasps:
    #         p = checker.collision_prob(obj, g, key=self.key, num=10)
    #         probs.append(p)
    #     self.assertEqual(len(probs), len(grasps))
    #     for p in probs:
    #         self.assertGreaterEqual(p, 0.0)
    #         self.assertLessEqual(p, 1.0)
