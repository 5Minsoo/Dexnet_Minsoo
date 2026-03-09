from Minsoo_net.grasp.contact import Contact3D
from Minsoo_net.grasp.Object import GraspableObject3D
from Minsoo_net.grasp.grasp import ParallelJawGrasp
from Minsoo_net.grasp.grasp_sampler import AntipodalGraspSampler
from Minsoo_net.grasp.collision_checker import GraspCollisionChecker
from Minsoo_net.grasp.gripper import RobotGripper
from Minsoo_net.grasp.visualize import visualize_grasps
import numpy as np

config_path='/home/minsoo/bin_picking1/gripper_config.yaml'
gripper_yaml='/home/minsoo/bin_picking1/Minsoo_net/data/gripper/gripper.yaml'
graspable1=GraspableObject3D('object.stl',sdf_resolution=256)
sampler=AntipodalGraspSampler(config_path)
gripper=RobotGripper('hande',gripper_yaml)

point=sampler.generate_grasps(graspable1,50)

checker=GraspCollisionChecker(gripper)

checker.set_table()

stable_poses, probs = graspable1.stable_poses()

# 가장 안정적인 pose 선택
best_idx = np.argmax(probs)
best_pose = stable_poses[best_idx]
checker.set_object('bin',graspable1.mesh,best_pose)
# grasp을 테이블에 정렬
aligned_grasps = [grasp.perpendicular_table(best_pose) for grasp in point]

# 충돌 검사: approach 각도 필터 + 충돌 체크
max_approach_angle = np.deg2rad(45)
approach_dist = 0.05
delta_approach = 0.005
collision_free_grasps = []

for grasp in aligned_grasps:
    _, approach_angle, _ = grasp.grasp_angles_from_stp_z(best_pose)

    if not checker.in_collision(visualize=True):
        collision_free_grasps.append(grasp)

print(f"Total grasps: {len(point)}")
print(f"Collision-free grasps: {len(collision_free_grasps)}")

visualize_grasps(graspable1, collision_free_grasps)