from Minsoo_net.grasp import (
    Contact3D, 
    GraspableObject3D, 
    ParallelJawGrasp, 
    AntipodalGraspSampler,
    GraspCollisionChecker,
    RobotGripper,
    visualize_grasps,
    PointGraspMetrics3D,
    QuasiStaticGraspQualityRV,
    GraspableObjectPoseGaussianRV,ParallelJawGraspPoseGaussianRV,ParamsGaussianRV
)
import numpy as np
import copy
import os
############## 파일상대경로 #################
base_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_dir,"gripper_config.yaml")
gripper_yaml=os.path.join(base_dir,"Minsoo_net","data","gripper","gripper.yaml")
rv_config_path=os.path.join(base_dir,"Minsoo_net","config","random_variables.yaml")
config_path = os.path.normpath(config_path)
gripper_yaml=os.path.normpath(gripper_yaml)
###########################################

graspable1=GraspableObject3D('object.stl',sdf_resolution=256)
sampler=AntipodalGraspSampler(config_path)
gripper=RobotGripper('hande',gripper_yaml)
checker=GraspCollisionChecker(gripper)
point=sampler.generate_grasps(graspable1,3)


stable_poses, probs = graspable1.stable_poses()
# 가장 안정적인 pose 선택
# grasp을 테이블에 정렬


# 충돌 검사: approach 각도 필터 + 충돌 체크
max_approach_angle = np.deg2rad(30)
approach_dist = 0.05
delta_approach = 0.005
prob_threshold=0.012

for idx, (pose, prob) in enumerate(zip(stable_poses, probs)):
    collision_free_grasps = []
    quality_grasps=[]
    quality=[]
    if prob < prob_threshold:
        continue
    checker.set_table() 
    best_pose = stable_poses[idx]
    checker.set_object(f'bin_{idx}',graspable1.mesh,T_world_obj=best_pose)
    aligned_grasps = [grasp.perpendicular_table(best_pose) for grasp in point]

    for grasp in aligned_grasps: 
        _, approach_angle, _ = grasp.grasp_angles_from_stp_z(best_pose)
        if approach_angle < max_approach_angle:
            collision_free_grasps.append(grasp)
            continue
        if not checker.grasp_in_collision(grasp.T_grasp_obj, key=f'bin_{idx}'):
            collision_free_grasps.append(grasp)

    print(f"Total grasps: {len(point)}")
    print(f"Collision-free grasps: {len(collision_free_grasps)}")

    i=1
    for grasp in (collision_free_grasps):

        obj_rv=GraspableObjectPoseGaussianRV(mean_T_obj_world=stable_poses,obj=graspable1,config_yaml=rv_config_path)
        grasp_rv=ParallelJawGraspPoseGaussianRV(grasp,rv_config_path)
        friction_rv=ParamsGaussianRV(rv_config_path)
        q=QuasiStaticGraspQualityRV(grasp_rv,obj_rv,friction_rv)
        avg_quality=q.expected_quality(100)
        print(f"Grasp {i}: Quality = {avg_quality:.4f}")
        i+=1
        # 임계값(0.03) 이상인 것만 필터링
        if avg_quality >= 0.03:
            quality_grasps.append(grasp)
    checker.remove_object(f'bin_{idx}')
    print('collision free grasp 개수:',len(collision_free_grasps))
    print('quality grasp 개수:',len(quality_grasps))
    visualize_grasps(graspable=graspable1, grasps=collision_free_grasps, pose=best_pose,gripper=gripper)