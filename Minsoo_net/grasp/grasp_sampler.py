# -*- coding: utf-8 -*-
"""
Antipodal grasp sampler for parallel-jaw grippers.
Based on Dex-Net (UC Berkeley).
Minsoo_net 클래스 사용 버전.
"""
import copy
import logging
import random
import yaml
import numpy as np

from Minsoo_net.grasp.contact import Contact3D
from Minsoo_net.grasp.grasp import ParallelJawGrasp


class AntipodalGraspSampler:
    """
    마찰원뿔 기반 antipodal rejection sampling으로 grasp를 생성한다.

    1. 표면점 하나를 선택
    2. 마찰원뿔 내에서 방향을 샘플링
    3. 반대편 접촉점을 찾아 force closure 검사
    4. 통과하면 유효한 grasp로 저장

    Parameters
    ----------
    config  : dict / YamlConfig
    """

    def __init__(self, config_path):
        with open(config_path,'r') as f:
            config=yaml.safe_load(f)

        # --- config ---
        self.max_width = config['gripper_max_width']
        self.friction_coef = config['sampling_friction_coef']
        self.num_cone_faces = config['num_cone_faces']
        self.num_samples = config['grasp_samples_per_surface_point']
        self.target_num_grasps = config.get('target_num_grasps') or config['min_num_grasps']
        self.num_grasp_rots = config['num_grasp_rots']
        self.max_num_surface_points = config.get('max_num_surface_points', 100)
        self.grasp_dist_thresh = config.get('grasp_dist_thresh', 0)
        self.max_iter=config.get('max_sampling_iteration',100)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def generate_grasps(
        self,
        graspable,
        target_num_grasps=None,
        grasp_gen_mult=5,
        max_iter=100,
        sample_approach_angles=False,
    ):
        """
        목표 개수만큼 antipodal grasp를 생성한다.

        내부적으로 sample_grasps → 거리 중복 제거 → (옵션) 회전 확장
        을 목표 개수에 도달하거나 max_iter까지 반복한다.
        """
        max_iter=self.max_iter
        if target_num_grasps is None:
            target_num_grasps = self.target_num_grasps

        grasps = []
        remaining = target_num_grasps
        k = 1

        while remaining > 0 and k <= max_iter:
            new_grasps = self.sample_grasps(graspable, grasp_gen_mult * remaining)

            # 거리 기반 중복 제거
            pruned = []
            for g in new_grasps:
                min_dist = min(
                    (np.linalg.norm(eg.center - g.center) for eg in (*grasps, *pruned)),
                    default=np.inf,
                )
                if min_dist >= self.grasp_dist_thresh:
                    pruned.append(g)

            # 접근 각도 회전 확장
            if sample_approach_angles:
                expanded = []
                delta_theta = 2 * np.pi / self.num_grasp_rots
                for g in pruned:
                    for i in range(self.num_grasp_rots):
                        rg = copy.copy(g)
                        rg.approach = self._approach_from_angle(g.axis, i * delta_theta)
                        expanded.append(rg)
                pruned = expanded

            grasps += pruned
            print(f'{len(grasps)}/{target_num_grasps} grasps after iter {k}')

            grasp_gen_mult *= 2
            remaining = target_num_grasps - len(grasps)
            k += 1

        random.shuffle(grasps)
        grasps = grasps[:target_num_grasps]
        print(f'Final: {len(grasps)} grasps')
        return grasps

    def sample_grasps(self, graspable, num_grasps):
        """
        Antipodal rejection sampling 핵심 루프.

        표면점마다 마찰원뿔 내 방향을 뽑고,
        반대편 접촉의 force closure를 확인한다.
        """
        grasps = []

        surface_points, _ = graspable.sample_surface(self.max_num_surface_points)
        np.random.shuffle(surface_points)

        for x_surf in surface_points:
            for _ in range(self.num_samples):
                # 1) 표면점 근처 섭동
                x1 = self._perturb_point(x_surf, graspable.surface_thresh)
                
                # 2) 마찰원뿔 계산
                c1 = Contact3D(graspable, x1, in_direction=None)
                in_norm, tx1, ty1 = c1.tangents()
                cone_ok, cone1, n1 = c1.friction_cone(self.num_cone_faces, self.friction_coef)
                if not cone_ok:
                    continue

                # 3) 원뿔 내 방향 샘플링
                v = self._sample_from_cone(n1, tx1, ty1)

                # 4) ParallelJawGrasp 생성
                center = x1 + (self.max_width / 2.0) * v
                grasp = ParallelJawGrasp(
                    center=center,
                    axis=v,
                    open_width=self.max_width,
                )

                # 5) close_fingers로 정확한 접촉 재계산
                success, contacts = grasp.close_fingers(graspable, check_approach=False)
                if not success:
                    continue
                c1_true, c2_true = contacts[0], contacts[1]
                center=(c1_true.point+c2_true.point)*0.5
                grasp = ParallelJawGrasp(
                    center=center,
                    axis=v,
                    open_width=self.max_width,
                    contact_points=[c1_true, c2_true],
                )

                # 7) 반대편 마찰원뿔 + force closure
                cone_ok2, cone2, n2 = c2_true.friction_cone(self.num_cone_faces, self.friction_coef)
                if not cone_ok2:
                    continue

                if self._force_closure(c1_true, c2_true, self.friction_coef):
                    grasps.append(grasp)

            # 조기 종료
            if len(grasps) >= num_grasps:
                break

        random.shuffle(grasps)
        return grasps

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _perturb_point(x, scale):
        """균일 랜덤 섭동."""
        return x + (scale / 2.0) * (np.random.rand(3) - 0.5)

    def _sample_from_cone(self, n, tx, ty):
        """마찰원뿔 내에서 방향 하나를 균일 샘플링."""
        theta = 2 * np.pi * np.random.rand()
        r = self.friction_coef * np.random.rand()
        v = n + r * np.cos(theta) * tx + r * np.sin(theta) * ty
        return -v / np.linalg.norm(v)

    @staticmethod
    def _force_closure(c1, c2, friction_coef):
        """두 접촉점의 force closure 검사 (마찰원뿔 기반)."""
        if c1.normal is None or c2.normal is None:
            return False
        v = c2.point - c1.point
        dist = np.linalg.norm(v)
        if dist < 1e-8:
            return False
        v = v / dist
        cos_thresh = 1.0 / np.sqrt(1.0 + friction_coef ** 2)
        return (np.dot(v, -c1.normal) >= cos_thresh and
                np.dot(v, c2.normal) >= cos_thresh)

    @staticmethod
    def _approach_from_angle(axis, theta):
        """grasp axis에 직교하는 평면에서 theta 각도의 approach 벡터 생성."""
        ref = np.array([0, 1, 0]) if abs(axis[0]) >= 0.9 else np.array([1, 0, 0])
        u = np.cross(axis, ref)
        u /= np.linalg.norm(u)
        w = np.cross(axis, u)
        return np.cos(theta) * u + np.sin(theta) * w