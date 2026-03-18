import numpy as np
import copy
from pathlib import Path
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
    GraspableObjectPoseGaussianRV,
    ParallelJawGraspPoseGaussianRV,
    ParamsGaussianRV
)

p= Path(__file__).parent.parent.resolve()
class GraspPipeline:
    def __init__(self, 
                 obj_file= None, 
                 num_grasps=3, 
                 prob_threshold=0.012, 
                 quality_threshold=0.03,
                 max_approach_angle_deg=30):
        
        # 1. 파일 경로 설정
        self.config_path = p / "config" / "gripper_config.yaml"
        self.gripper_yaml = p /"data"/ "gripper" / "gripper.yaml"
        self.rv_config_path = p/ "config" / "random_variables.yaml"
        
        # 2. 파라미터 초기화
        self.num_grasps = num_grasps
        self.prob_threshold = prob_threshold
        self.quality_threshold = quality_threshold
        self.max_approach_angle = np.deg2rad(max_approach_angle_deg)
        
        # 3. 핵심 컴포넌트 로드
        self.graspable_obj = GraspableObject3D(obj_file, sdf_resolution=256)    
        self.sampler = AntipodalGraspSampler(self.config_path)
        self.gripper = RobotGripper('hande', self.gripper_yaml)
        self.checker = GraspCollisionChecker(self.gripper)

        self.initial_grasps = self.sampler.generate_grasps(self.graspable_obj, self.num_grasps)
        
        # 안정적인 Pose 계산
        self.stable_poses, self.probs = self.graspable_obj.stable_poses()

    def filter_collision_free_grasps(self, aligned_grasps, pose, obj_key):
        """접근 각도 필터링 및 충돌 검사를 수행합니다."""
        collision_free_grasps = []
        for grasp in aligned_grasps:
            _, approach_angle, _ = grasp.grasp_angles_from_stp_z(pose)
            
            if approach_angle < self.max_approach_angle:
                collision_free_grasps.append(grasp)
                continue
                
            if not self.checker.grasp_in_collision(grasp.T_grasp_obj, key=obj_key):
                collision_free_grasps.append(grasp)
                
        return collision_free_grasps

    def evaluate_grasp_quality(self, collision_free_grasps, num_samples=100):
        """Grasp의 Quality를 확률적으로 평가합니다."""
        quality_grasps = []
        quality=[]
        print(f"{self.quality_threshold} 이하 grasp 폐기 ")
        for i, grasp in enumerate(collision_free_grasps, start=1):
            # 기존 코드에 맞춰 stable_poses를 전달합니다 (필요에 따라 현재 pose로 변경 가능)
            obj_rv = GraspableObjectPoseGaussianRV(obj=self.graspable_obj, config_yaml=self.rv_config_path)
            grasp_rv = ParallelJawGraspPoseGaussianRV(grasp, self.rv_config_path)
            friction_rv = ParamsGaussianRV(self.rv_config_path)
            
            q = QuasiStaticGraspQualityRV(grasp_rv, obj_rv, friction_rv)
            avg_quality = q.expected_quality(num_samples)
            
            print(f"Grasp {i}: Quality = {avg_quality:.4f}")

            if avg_quality >= self.quality_threshold:
                quality_grasps.append(grasp)
                quality.append(avg_quality)
                
        return quality_grasps,quality

    def execute(self,use_visual=False,start_index=0):
        """전체 Grasp 생성 및 필터링 파이프라인을 실행합니다.
        returns stable_poses,initial_grasps,collision_free,quality_grasps,qualities
        """
        yielded_count = 0
        for idx, (pose, prob) in enumerate(zip(self.stable_poses, self.probs)):
            if prob < self.prob_threshold:
                continue
                
            if yielded_count < start_index:
                print(f"  -> 유효한 Pose 스킵 (현재 {yielded_count}/{start_index})")
                yielded_count += 1
                continue
            print(f"\n>>> [RUN] 원본 stable_poses[{idx}] -> 현재 계산 중 (yielded_count: {yielded_count})")
            obj_key = f'bin_{idx}'
            self.checker.set_table() 
            self.checker.set_object(obj_key, self.graspable_obj.mesh, T_world_obj=pose)
            
            # 테이블에 맞춰 정렬
            aligned_grasps = [grasp.perpendicular_table(pose) for grasp in self.initial_grasps]
            
            # 1. 충돌 검사
            collision_free_grasps = self.filter_collision_free_grasps(aligned_grasps, pose, obj_key)
            
            print(f"Total grasps: {len(self.initial_grasps)}")
            print(f"Collision-free grasps: {len(collision_free_grasps)}")
            
            # 2. Quality 평가
            quality_grasps,quality = self.evaluate_grasp_quality(collision_free_grasps)
            
            self.checker.remove_object(obj_key)
            
            print(f'collision free grasp 개수: {len(collision_free_grasps)}')
            print(f'quality grasp 개수: {len(quality_grasps)}')
            if use_visual:
                visualize_grasps(self.graspable_obj,collision_free_grasps,pose=pose,gripper=self.gripper)
                # collision_free_grasps=[grasp.transform(pose) for grasp in collision_free_grasps]
                # quality_grasps = [grasp.transform(pose) for grasp in quality_grasps]
                visualize_grasps(self.graspable_obj,quality_grasps,pose=pose,gripper=self.gripper)
            yield pose, collision_free_grasps,quality_grasps, quality
            yielded_count += 1


# === 실행 예시 ===
if __name__ == "__main__":
    # 클래스 초기화 시 파라미터를 유연하게 변경 가능합니다.
    pipeline = GraspPipeline(
        obj_file='/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/object.stl',
        num_grasps=10,
        prob_threshold=0.012,
        quality_threshold=0.03
    )
    
    # 파이프라인 실행
    pipeline.execute()