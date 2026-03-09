# -*- coding: utf-8 -*-
"""
Quasi-static point-based grasp quality metrics (simplified).
Metrics: grasp_matrix, force_closure (Nguyen 1988), ferrari_canny_L1.
Hard-finger contact model only.

Original authors: Jeff Mahler and Brian Hou
Based on code from UC Berkeley (BSD-like license).
Minsoo_net Contact3D 클래스 사용 버전.
"""
import logging
import sys

import numpy as np

from contact import Contact3D

try:
    import pyhull.convex_hull as cvh
except ImportError:
    logging.warning("Failed to import pyhull")

try:
    import cvxopt as cvx
    cvx.solvers.options["show_progress"] = False
except ImportError:
    logging.warning("Failed to import cvxopt")


class PointGraspMetrics3D:
    """Quasi-static quality metrics for two-finger point grasps (hard finger).

    Minsoo_net의 Contact3D 객체를 직접 받아 품질 메트릭을 계산한다.
    """

    @staticmethod
    def grasp_matrix(forces, torques, torque_scaling=1.0):
        """Build the 6×N grasp map from contact forces/torques to object wrenches.

        Parameters
        ----------
        forces : ndarray, shape (3, N)
        torques : ndarray, shape (3, N)
        torque_scaling : float

        Returns
        -------
        G : ndarray, shape (6, N)
        """
        if forces.shape[1] != torques.shape[1]:
            raise ValueError("Need same number of forces and torques")

        n = forces.shape[1]
        G = np.zeros([6, n])
        G[:3, :] = forces
        G[3:, :] = torque_scaling * torques
        return G

    @staticmethod
    def force_closure(c1: Contact3D, c2: Contact3D, friction_coef, use_abs_value=True):
        """Fast two-contact force-closure test via antipodality (Nguyen 1988).

        Parameters
        ----------
        c1, c2 : Contact3D
        friction_coef : float
        use_abs_value : bool

        Returns
        -------
        int : 1 if force closure, 0 otherwise.
        """
        if (c1.point is None or c2.point is None
                or c1.normal is None or c2.normal is None):
            return 0

        p1, p2 = c1.point, c2.point
        n1, n2 = -c1.normal, -c2.normal

        if np.allclose(p1, p2):
            return 0

        for normal, contact, other in [(n1, p1, p2), (n2, p2, p1)]:
            diff = other - contact
            proj = normal.dot(diff) / np.linalg.norm(normal)
            if use_abs_value:
                proj = abs(proj)
            if proj < 0:
                return 0
            alpha = np.arccos(proj / np.linalg.norm(diff))
            if alpha > np.arctan(friction_coef):
                return 0
        return 1

    @staticmethod
    def ferrari_canny_L1(forces, torques, torque_scaling=1.0,
                         wrench_norm_thresh=1e-3, wrench_regularizer=1e-10):
        """Ferrari & Canny's L1 (epsilon) metric (raw arrays).

        Parameters
        ----------
        forces : ndarray, shape (3, N)
        torques : ndarray, shape (3, N)
        torque_scaling : float
        wrench_norm_thresh : float
        wrench_regularizer : float

        Returns
        -------
        float : epsilon value (0 means no force closure).
        """
        G = PointGraspMetrics3D.grasp_matrix(forces, torques, torque_scaling)

        hull = cvh.ConvexHull(G.T)
        if len(hull.vertices) == 0:
            return 0.0

        min_norm, v = PointGraspMetrics3D._min_norm_vector_in_facet(
            G, wrench_regularizer
        )
        if min_norm > wrench_norm_thresh:
            return 0.0
        if np.sum(v > 1e-4) <= G.shape[0] - 1:
            return 0.0

        min_dist = sys.float_info.max
        for vertex_indices in hull.vertices:
            if np.max(np.array(vertex_indices)) < G.shape[1]:
                facet = G[:, vertex_indices]
                dist, _ = PointGraspMetrics3D._min_norm_vector_in_facet(
                    facet, wrench_regularizer
                )
                min_dist = min(min_dist, dist)

        return min_dist

    @staticmethod
    def ferrari_canny_L1_from_contacts(
        c1: Contact3D,
        c2: Contact3D,
        friction_coef: float = 0.5,
        num_cone_faces: int = 8,
        torque_scaling: float = 1.0,
        wrench_norm_thresh: float = 1e-3,
        wrench_regularizer: float = 1e-10,
    ) -> float:
        """두 Contact3D 객체로부터 Ferrari-Canny L1 품질을 계산한다.

        Contact3D의 friction_cone()과 torques()를 사용하여
        forces/torques를 자동으로 추출한 뒤 ferrari_canny_L1을 호출한다.

        Parameters
        ----------
        c1, c2 : Contact3D
        friction_coef : float
        num_cone_faces : int
        torque_scaling : float
        wrench_norm_thresh : float
        wrench_regularizer : float

        Returns
        -------
        float : epsilon value (0 means no force closure).
        """
        # 마찰원뿔에서 힘 벡터 추출
        ok1, cone1, _ = c1.friction_cone(num_cone_faces, friction_coef)
        ok2, cone2, _ = c2.friction_cone(num_cone_faces, friction_coef)
        if not ok1 or not ok2:
            return 0.0

        # 합쳐진 힘 행렬 (3, 2*num_cone_faces)
        forces = np.hstack([cone1, cone2])

        # 각 접촉점에서 토크 계산
        ok_t1, torques1 = c1.torques(cone1)
        ok_t2, torques2 = c2.torques(cone2)
        if not ok_t1 or not ok_t2:
            return 0.0

        torques = np.hstack([torques1, torques2])

        return PointGraspMetrics3D.ferrari_canny_L1(
            forces, torques, torque_scaling,
            wrench_norm_thresh, wrench_regularizer,
        )

    @staticmethod
    def _min_norm_vector_in_facet(facet, wrench_regularizer=1e-10):
        """Min-norm point in convex hull of facet columns via QP.

        Solves:  min ‖facet · α‖²  s.t.  α ≥ 0, Σα = 1
        """
        dim = facet.shape[1]
        gram = facet.T.dot(facet) + wrench_regularizer * np.eye(dim)

        P = cvx.matrix(2 * gram)
        q = cvx.matrix(np.zeros((dim, 1)))
        G_ineq = cvx.matrix(-np.eye(dim))
        h = cvx.matrix(np.zeros((dim, 1)))
        A = cvx.matrix(np.ones((1, dim)))
        b = cvx.matrix(np.ones(1))

        sol = cvx.solvers.qp(P, q, G_ineq, h, A, b)
        v = np.array(sol["x"])
        min_norm = np.sqrt(sol["primal objective"])

        return abs(min_norm), v