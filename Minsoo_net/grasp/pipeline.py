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
                 max_approach_angle_deg=15,
                 num_poses=10):
        
        # 1. 파일 경로 설정
        self.config_path = p / "config" / "master_config.yaml"
        self.gripper_yaml = p / "config" / "master_config.yaml"
        self.rv_config_path = p / "config" / "master_config.yaml"
        
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
        self.num_poses=num_poses
    def filter_collision_free_grasps(self, aligned_grasps, pose, obj_key):
        """접근 각도 필터링 및 충돌 검사를 수행합니다."""
        collision_free_grasps = []
        collision_grasps = []        
        for grasp in aligned_grasps:
            # _, approach_angle, _ = grasp.grasp_angles_from_stp_z(pose)
            
            # if approach_angle > self.max_approach_angle:
            #     continue
            
            #######################collide along approach로 변경 가능####################
            # if not self.checker.grasp_in_collision(grasp.T_grasp_obj, key=obj_key):
            #     collision_free_grasps.append(grasp)

            if not self.checker.collides_along_approach(grasp, key=obj_key):
                collision_free_grasps.append(grasp)
            else:
                collision_grasps.append(grasp)
        return collision_grasps,collision_free_grasps

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
        for idx, (pose, prob) in enumerate(zip(self.stable_poses[:self.num_poses], self.probs[:self.num_poses])):
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
            valid_grasps = []
            # 테이블에 맞춰 정렬
            perpendicular_grasps = [grasp.perpendicular_table(pose) for grasp in self.initial_grasps]
            for grasp in perpendicular_grasps:
                _, approach_angle, _ = grasp.grasp_angles_from_stp_z(pose)
                
                if approach_angle < self.max_approach_angle:
                    valid_grasps.append(grasp)

            aligned_grasps = [g for grasp in valid_grasps for g in grasp.multi_angle_grasps(pose)]
            # 1. 충돌 검사
            collision_grasps, collision_free_grasps = self.filter_collision_free_grasps(aligned_grasps, pose, obj_key)
            
            print(f"Total grasps: {len(self.initial_grasps)}")
            print(f"Collision-free grasps: {len(collision_free_grasps)}")
            print(f"Collision grasps: {len(collision_grasps)}")
            seen={}
            for grasp in collision_free_grasps:
                key=grasp.grasp_key()
                if key not in seen:
                    seen[key]=grasp
            unique_grasps = list(seen.values())
            
            # 2. Quality 평가
            quality_grasps, quality = self.evaluate_grasp_quality(unique_grasps)

            quality_map = {g.grasp_key(): q for g, q in zip(quality_grasps, quality)}

            quality_grasps = [g for g in collision_free_grasps if g.grasp_key() in quality_map]
            quality = [quality_map[g.grasp_key()] for g in quality_grasps]
            # 3. Failed Grasps 병합 (충돌 발생 + Quality 미달)
            # evaluate_grasp_quality 구현에 따라 다르겠지만, 일반적인 리스트 필터링 기준입니다.
            low_quality_grasps = [g for g in collision_free_grasps if g.grasp_key() not in quality_map]
            failed_grasps = collision_grasps + low_quality_grasps

            
            self.checker.remove_object(obj_key)
            
            print(f'failed grasp 개수: {len(failed_grasps)}') # 추가된 로그
            print(f'quality grasp 개수: {len(quality_grasps)}')
            
            if use_visual:
                visualize_grasps(self.graspable_obj, collision_grasps, pose=pose, gripper=self.gripper,title="Collision Grasps")
                visualize_grasps(self.graspable_obj, collision_free_grasps, pose=pose, gripper=self.gripper,title="Collision free Grasps")
                visualize_grasps(self.graspable_obj, quality_grasps, pose=pose, gripper=self.gripper,title="Quality Grasps")
            
            yield pose, failed_grasps, quality_grasps, quality
            yielded_count += 1


if __name__ == "__main__":
    # 1. 설정 파라미터
    # 실제 환경에 맞춰 경로를 수정하세요.
    OBJ_FILE_PATH = '/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/object/PVCTT13.stl'
    OBJ_FILE_PATH='/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/object/bin.stl'
    
    # --- 테스트 제어 변수 ---
    START_INDEX = 1      # 0부터 시작하거나, 특정 Pose부터 재개하고 싶을 때 변경
    NUM_TEST_GRASPS = 300 # 한 Pose당 생성할 Grasp 후보 개수
    VISUALIZE = False   # 시각화 여부
    # -----------------------

    print(f"\n" + "="*60)
    print(f"🚀 Grasp 파이프라인 테스트를 시작합니다. (시작 인덱스: {START_INDEX})")
    print(f"📦 대상 파일: {OBJ_FILE_PATH}")
    print("="*60)

    # 2. 파이프라인 초기화
    try:
        pipeline = GraspPipeline(
            obj_file=OBJ_FILE_PATH,
            num_grasps=NUM_TEST_GRASPS,
            prob_threshold=0.012,
            quality_threshold=0.002
        )
        
        print(f"✅ 객체 로드 및 Stable Poses 계산 완료. (총 {len(pipeline.stable_poses)}개 발견)")

        # 3. 파이프라인 실행 (Generator 순회)
        # execute 메서드에 start_index를 전달하여 내부 스킵 로직 작동
        results_generator = pipeline.execute(use_visual=VISUALIZE, start_index=START_INDEX)

        for i, (pose, failed_grasps, quality_grasps, qualities) in enumerate(results_generator):
            current_pose_idx = START_INDEX + i
            
            print(f"\n[결과 리포트 - Pose {current_pose_idx}]")
            print(f"  - 성공(Quality) Grasp: {len(quality_grasps)}개, Quality개수: {len(qualities)}")
            print(f"  - 실패(Collision/Low Quality) Grasp: {len(failed_grasps)}개")
            
            if len(qualities) > 0:
                print(f"  - 최고 품질 점수: {max(qualities):.4f}")
                print(f"  - 평균 품질 점수: {np.mean(qualities):.4f}")
            else:
                print("  - ⚠️ 이 Pose에서는 유효한 Grasp을 찾지 못했습니다.")

            print("-" * 40)
            # 시각화가 켜져 있을 경우, 창을 닫아야 다음 Pose로 넘어갑니다.
            if VISUALIZE:
                print(f"💡 Pose {current_pose_idx} 시각화 중... 다음으로 넘어가려면 창을 닫으세요.")

    except FileNotFoundError:
        print(f"❌ 오류: 파일을 찾을 수 없습니다. 경로를 확인하세요: {OBJ_FILE_PATH}")
    except Exception as e:
        print(f"❌ 실행 중 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()

    print("\n[🏁 테스트 종료] 모든 처리가 완료되었습니다.")