import numpy as np
from typing import Optional, Tuple, List, Dict
from Minsoo_net.grasp.contact import Contact3D
import yaml
from copy import deepcopy
class ParallelJawGrasp:
    """
    평행 조 그리퍼 grasp 하나를 표현.
    SDF 기반으로 손가락 닫힘 시뮬레이션을 수행한다.
    """

    def __init__(
        self,
        center: np.ndarray,
        axis: np.ndarray,
        open_width: float = 0.05,
        close_width: float = 0.0,
        approach: Optional[np.ndarray] = None,
        samples_per_unit: float = 500.0,
    ):
        self.center = np.array(center, dtype=np.float64)
        self.axis = np.array(axis, dtype=np.float64)
        self.axis /= np.linalg.norm(self.axis)
        self.open_width = open_width
        self.close_width = close_width
        self.samples_per_unit = samples_per_unit
        self.approach_angle=None
        self.T_grasp_obj=np.eye(4)
        if approach is not None:
            self.approach = np.array(approach, dtype=np.float64)
            self.approach /= np.linalg.norm(self.approach)

    @property
    def endpoints(self) -> Tuple[np.ndarray, np.ndarray]:
        half = self.open_width / 2.0
        return self.center - half * self.axis, self.center + half * self.axis

    def close_fingers(
        self,
        obj,
        check_approach: bool = True,
        approach_dist: float = 0.05,
    ) -> Tuple[bool, Optional[List[Dict]]]:
        """
        손가락을 닫으며 접촉점 2개를 찾는다.

        Returns
        -------
        success  : 양쪽 모두 접촉 찾았는지
        contacts : [{'point', 'normal', 'in_direction'}, ...] 또는 None
        """
        g1, g2 = self.endpoints
        num_samples = max(int(self.samples_per_unit * self.open_width / 2), 3)

        # 접근 경로 충돌 검사
        if check_approach:
            n_app = max(int(self.samples_per_unit * approach_dist / 2), 3)
            loa_app1 = self.create_line_of_action(g1, -self.approach, approach_dist, n_app)
            loa_app2 = self.create_line_of_action(g2, -self.approach, approach_dist, n_app)
            if self.find_contact(loa_app1, obj)[0] or self.find_contact(loa_app2, obj)[0]:
                return False, None

        # 손가락 닫힘 경로
        loa1 = self.create_line_of_action(g1, self.axis, self.open_width, num_samples)
        loa2 = self.create_line_of_action(g2, -self.axis, self.open_width, num_samples)

        # 접촉점 탐색
        c1_found, c1 = self.find_contact(loa1, obj)
        c2_found, c2 = self.find_contact(loa2, obj)

        if c1_found and c2_found:
            return True, [c1, c2]
        return False, None

    def create_line_of_action(
        self,
        start: np.ndarray,
        direction: np.ndarray,
        width: float,
        num_samples: int
    ) -> List[np.ndarray]:
        num_samples = max(num_samples, 3)
        travel = width / 2.0
        return [start + t * direction for t in np.linspace(0, travel, num=num_samples)]

    def find_contact(
        self,
        line_of_action: List[np.ndarray],
        obj,
    ) -> Tuple[bool, Optional['Contact3D']]:
        sdf_vals = np.array([obj.sdf(pt).item() for pt in line_of_action])
        n = len(line_of_action)

        for i in range(n):
            on_surface = np.abs(sdf_vals[i]) < obj.surface_thresh
            if not on_surface and i > 0:
                on_surface = sdf_vals[i - 1] * sdf_vals[i] < 0

            if on_surface:
                pt = None

                if i > 0 and sdf_vals[i - 1] * sdf_vals[i] < 0:
                    if i + 1 < n:
                        pt = obj.find_zero_crossing_quadratic(
                            line_of_action[i - 1], sdf_vals[i - 1],
                            line_of_action[i],     sdf_vals[i],
                            line_of_action[i + 1], sdf_vals[i + 1],
                        )
                    if pt is None:
                        t = sdf_vals[i - 1] / (sdf_vals[i - 1] - sdf_vals[i])
                        pt = line_of_action[i - 1] + t * (line_of_action[i] - line_of_action[i - 1])
                else:
                    pt = line_of_action[i]

                normal = obj.surface_normal(pt)
                if np.linalg.norm(normal) < 1e-8:
                    continue

                in_dir = line_of_action[-1] - line_of_action[0]
                in_dir /= np.linalg.norm(in_dir)

                return True, Contact3D(obj, pt, in_direction=in_dir)

        return False, None

    def __repr__(self):
        return (
            f"ParallelJawGrasp(center={self.center}, "
            f"axis={self.axis}, width={self.open_width:.4f})"
        )
    
    def grasp_angles_from_stp_z(self, stable_pose):
        """ Get angles of the the grasp from the table plane:
        1) the angle between the grasp axis and table normal
        2) the angle between the grasp approach axis and the table normal
        
        Parameters
        ----------
        stable_pose : :obj:`StablePose` or :obj:`RigidTransform`
            the stable pose to compute the angles for

        Returns
        -------
        psi : float
            grasp y axis rotation from z axis in stable pose
        phi : float
            grasp x axis rotation from z axis in stable pose
        """
        T_grasp_obj = self.T_grasp_obj        
        T_stp_obj = stable_pose
        T_stp_grasp = T_stp_obj * T_grasp_obj

        stp_z = np.array([0,0,1])
        grasp_axis_angle = np.arccos(stp_z.dot(T_stp_grasp[:3,1]))
        grasp_approach_angle = np.arccos(abs(stp_z.dot(T_stp_grasp[:3,2])))
        nu = stp_z.dot(T_stp_grasp[:3,0])

        return grasp_axis_angle, grasp_approach_angle, nu

    def _angle_aligned_with_table(self, table_normal):
        """
        Returns the y-axis rotation angle that'd allow the current pose to align with the table normal.
        """
        def _argmax(f, a, b, n):
            #finds the argmax x of f(x) in the range [a, b) with n samples
            delta = (b - a) / n
            max_y = f(a)
            max_x = a
            for i in range(1, n):
                x = i * delta
                y = f(x)
                if y >= max_y:
                    max_y = y
                    max_x = x
            return max_x
    
        def _get_matrix_product_z_axis(grasp_axis, normal):
            def matrix_product(theta):
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                R = np.c_[[cos_t, 0, sin_t], np.c_[[0, 1, 0], [-sin_t, 0, cos_t]]]
                grasp_axis_rotated = np.dot(R, grasp_axis)
                return np.dot(normal, grasp_axis_rotated)
            return matrix_product
        
        axis=self.unrotated_full_axis()
        theta = _argmax(_get_matrix_product_z_axis(np.array([0,0,1]), np.dot(axis.T, -table_normal)), 0, 2*np.pi, 64)        
        return theta

    def perpendicular_table(self, stable_pose):
        """
        Returns a grasp with approach_angle set to be aligned width the table normal specified in the given stable pose.

        Parameters
        ----------
        stable_pose : SE(3) Transformation matrix

        Returns
        -------
        :obj:`ParallelJawPtGrasp3D`
            aligned grasp
        """
        table_normal = stable_pose[2,:3]
        theta = self._angle_aligned_with_table(table_normal)
        new_grasp = deepcopy(self)
        new_grasp.approach_angle = theta
        
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R = np.c_[[cos_t, 0, sin_t], np.c_[[0, 1, 0], [-sin_t, 0, cos_t]]]
        new_grasp.T_grasp_obj[:3,:3]=R
        new_grasp.T_grasp_obj[:3,3]=self.center.T

        return new_grasp

    def unrotated_full_axis(self):
        """ Rotation matrix from canonical grasp reference frame to object reference frame. Z axis points out of the gripper palm along the 0-degree approach direction, Y axis points between the jaws, and the X axis is orthogonal.

        Returns
        -------
        :obj:`numpy.ndarray`
            rotation matrix of grasp
        """
        grasp_axis_y = self.axis
        grasp_axis_z = np.array([grasp_axis_y[1], -grasp_axis_y[0], 0])
        if np.linalg.norm(grasp_axis_z) == 0:
            grasp_axis_z = np.array([0, 0, 1])
        grasp_axis_z = grasp_axis_z / np.linalg.norm(grasp_axis_z)
        grasp_axis_x = np.cross(grasp_axis_y, grasp_axis_z)

        R = np.c_[grasp_axis_x, np.c_[grasp_axis_y, grasp_axis_z]]
        return R