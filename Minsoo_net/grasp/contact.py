# -*- coding: utf-8 -*-
"""
Contact3D for Minsoo_net.
dex-net Contact3D를 기반으로, 월드 좌표 기반 GraspableObject3D에 맞게 재작성.
"""

import itertools as it
import logging
import numpy as np


class Contact3D:
    """3D 접촉점 모델.

    모든 좌표는 월드 좌표계 기준.
    """

    def __init__(self, graspable, contact_point, in_direction=None):
        self.graspable = graspable
        self.point = np.array(contact_point, dtype=np.float64)
        self.in_direction = in_direction
        self.friction_cone_ = None
        self.normal = None

        self._compute_normal()

    def _compute_normal(self):
        """접촉점에서 바깥 방향 법선을 계산한다 (SDF gradient 방향)."""
        on_surface, _ = self.graspable.on_surface(self.point)
        if not np.any(on_surface):
            logging.debug('Contact point not on surface')
            return

        normal = self.graspable.surface_normal(self.point)
        if np.linalg.norm(normal) < 1e-8:
            return

        self.normal = normal

    def tangents(self, direction=None, align_axes=True, max_samples=1000):
        """접촉점에서의 방향 벡터와 접선 벡터 2개를 반환한다.

        Returns
        -------
        direction, t1, t2 : 각각 (3,) numpy.ndarray 또는 None
        in_normal : 물체에 들어가는 방향 법선벡터
        """
        if self.normal is None:
            return None, None, None

        if direction is None:
            direction = -self.normal

        if np.dot(self.normal, direction) > 0:
            direction = -direction

        direction = direction.reshape((3, 1))
        U, _, _ = np.linalg.svd(direction)
        x, y = U[:, 1], U[:, 2]

        # 오른손 법칙
        if np.cross(x, y).dot(direction.ravel()) < 0:
            y = -y
        v, w = x, y

        # 물체 x축에 접선 축 정렬
        if align_axes:
            max_ip = 0
            max_theta = 0
            target = np.array([1, 0, 0])
            d_theta = 2 * np.pi / float(max_samples)
            for i in range(max_samples):
                theta = i * d_theta
                v_cand = np.cos(theta) * x + np.sin(theta) * y
                if v_cand.dot(target) > max_ip:
                    max_ip = v_cand.dot(target)
                    max_theta = theta

            v = np.cos(max_theta) * x + np.sin(max_theta) * y
            w = np.cross(direction.ravel(), v)

        return np.squeeze(direction), v, w

    def normal_force_magnitude(self):
        """법선 방향 힘 크기 (0~1)."""
        normal_force_mag = 1.0
        if self.in_direction is not None and self.normal is not None:
            in_dir_norm = self.in_direction / np.linalg.norm(self.in_direction)
            normal_force_mag = np.dot(in_dir_norm, -self.normal)
        return max(normal_force_mag, 0.0)

    def friction_cone(self, num_cone_faces=8, friction_coef=0.5):
        """마찰 원뿔 이산화.

        Returns
        -------
        success : bool
        cone_support : (3, num_cone_faces) numpy.ndarray
        normal : (3,) numpy.ndarray
        """
        if self.friction_cone_ is not None and self.normal is not None:
            return True, self.friction_cone_, self.normal

        in_normal, t1, t2 = self.tangents()
        if in_normal is None:
            return False, self.friction_cone_, self.normal

        # 미끄러짐 검사
        if self.in_direction is not None:
            in_dir_norm = self.in_direction / np.linalg.norm(self.in_direction)
            normal_force_mag = self.normal_force_magnitude()
            tan_force_mag = np.sqrt(np.dot(in_dir_norm, t1) ** 2 + np.dot(in_dir_norm, t2) ** 2)
            if friction_coef * normal_force_mag < tan_force_mag:
                logging.debug('Contact would slip')
                return False, self.friction_cone_, self.normal

        cone_support = np.zeros((3, num_cone_faces))
        for j in range(num_cone_faces):
            angle = 2 * np.pi * j / num_cone_faces
            tan_vec = t1 * np.cos(angle) + t2 * np.sin(angle)
            cone_support[:, j] = in_normal + friction_coef * tan_vec

        self.friction_cone_ = cone_support
        return True, self.friction_cone_, self.normal

    def torques(self, forces):
        """접촉점에서 힘 벡터들이 발생시키는 토크.

        Returns
        -------
        success : bool
        torques : (3, N) numpy.ndarray
        """
        on_surface, _ = self.graspable.on_surface(self.point)
        if not np.any(on_surface):
            return False, None

        num_forces = forces.shape[1]
        torques = np.zeros((3, num_forces))
        moment_arm = self.point - self.graspable.mesh.center_mass
        for i in range(num_forces):
            torques[:, i] = np.cross(moment_arm, forces[:, i])
        return True, torques

    @classmethod
    def from_dict(cls, graspable, contact_dict):
        """ParallelJawGrasp.find_contact()의 dict 결과로부터 생성."""
        obj = cls(graspable, contact_dict['point'], contact_dict['in_direction'])
        obj.normal = contact_dict['normal']
        return obj

    def __repr__(self):
        return f"Contact3D(point={self.point}, normal={self.normal})"
