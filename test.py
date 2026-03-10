from Minsoo_net.grasp.contact import Contact3D
from Minsoo_net.grasp.Object import GraspableObject3D
from Minsoo_net.grasp.grasp import ParallelJawGrasp
from Minsoo_net.grasp.grasp_sampler import AntipodalGraspSampler
from Minsoo_net.grasp.collision_checker import GraspCollisionChecker
from Minsoo_net.grasp.gripper import RobotGripper
from Minsoo_net.grasp.visualize import visualize_grasps
from Minsoo_net.grasp.quality import PointGraspMetrics3D
import numpy as np
import copy
import os
############## 파일상대경로 #################
base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_dir,"gripper_config.yaml")
gripper_yaml=os.path.join(base_dir,"Minsoo_net","data","gripper","gripper.yaml")
config_path = os.path.normpath(config_path)
gripper_yaml=os.path.normpath(gripper_yaml)
###########################################

graspable1=GraspableObject3D('object.stl',sdf_resolution=256)
sampler=AntipodalGraspSampler(config_path)
gripper=RobotGripper('hande',gripper_yaml)
checker=GraspCollisionChecker(gripper)

point=sampler.generate_grasps(graspable1,50)

checker.set_table()
stable_poses, probs = graspable1.stable_poses()

# 가장 안정적인 pose 선택
best_idx = np.argmax(probs)
best_pose = stable_poses[2]


checker.set_object('bin',graspable1.mesh,best_pose)
# grasp을 테이블에 정렬
aligned_grasps = [grasp.perpendicular_table(best_pose) for grasp in point]

# 충돌 검사: approach 각도 필터 + 충돌 체크
max_approach_angle = np.deg2rad(30)
approach_dist = 0.05
delta_approach = 0.005
collision_free_grasps = []
quality_grasps=[]
quality=[]
for grasp in aligned_grasps:
    _, approach_angle, _ = grasp.grasp_angles_from_stp_z(best_pose)
    if approach_angle < max_approach_angle:
        collision_free_grasps.append(grasp)
        continue
    if not checker.grasp_in_collision(grasp.T_grasp_obj, key='bin'):
        collision_free_grasps.append(grasp)

print(f"Total grasps: {len(point)}")
print(f"Collision-free grasps: {len(collision_free_grasps)}")

final_quality_results = [] # 각 파지의 최종 점수를 저장할 리스트

for i, grasp in enumerate(collision_free_grasps):
    # voxel_size를 이용한 scaling은 적절합니다.
    length = 1.0 / graspable1.voxel_size
    c1, c2 = grasp.contact_points
    original_c1_pos = copy.deepcopy(c1.point)
    original_c2_pos = copy.deepcopy(c2.point)
    current_grasp_qualities = []
    # 마찰 계수 범위를 0.1 이상으로 설정하는 것이 안정적입니다.
    for j in np.linspace(0.5, 5.0, 10): 
        original_c1_pos=original_c1_pos+np.random.normal(scale=0.002, size=3)  # 접촉점에 작은 노이즈 추가
        original_c2_pos=original_c2_pos+np.random.normal(scale=0.002, size=3)
        fric = j 
        # ferrari_canny 계산
        q = PointGraspMetrics3D.ferrari_canny_L1_from_contacts(
            c1, c2, torque_scaling=length, friction_coef=fric
        )
        current_grasp_qualities.append(q)
    
    # 해당 파지의 평균 품질 계산
    avg_quality = np.mean(current_grasp_qualities)
    
    # 임계값(0.18) 이상인 것만 필터링 (원래 코드에서는 < 0.18로 되어있는데 보통은 큰 것을 고릅니다)
    if avg_quality >= 0.18:
        print(f"Grasp {i}: Quality = {avg_quality:.4f}")
        quality_grasps.append(grasp)
visualize_grasps(graspable=graspable1, grasps=quality_grasps, pose=best_pose,gripper=gripper)