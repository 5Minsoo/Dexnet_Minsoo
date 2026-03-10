import trimesh
import numpy as np
from pysdf import SDF
from scipy.interpolate import RegularGridInterpolator

class GraspableObject3D:
    def __init__(self, mesh_path, sdf_resolution=64, surface_thresh=0.001):
        self.mesh = trimesh.load(mesh_path)
        self.mesh.apply_scale(0.001)
        self.surface_thresh = surface_thresh
        self.resolution = sdf_resolution
        self.center_mass=self.mesh.center_mass
        sdf_fn = SDF(self.mesh.vertices, self.mesh.faces)

        # 월드 좌표로 바로 그리드 생성
        margin = 0.05
        bmin, bmax = self.mesh.bounds[0] - margin, self.mesh.bounds[1] + margin
        lx = np.linspace(bmin[0], bmax[0], sdf_resolution)
        ly = np.linspace(bmin[1], bmax[1], sdf_resolution)
        lz = np.linspace(bmin[2], bmax[2], sdf_resolution)
        self.voxel_size=abs(lx[0]-lx[1])
        x, y, z = np.meshgrid(lx, ly, lz, indexing='ij')
        pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        sdf_vals = -sdf_fn(pts)
        self.sdf_grid = sdf_vals.reshape(sdf_resolution, sdf_resolution, sdf_resolution)

        # 보간기도 월드 좌표 기준
        self.interp = RegularGridInterpolator(
            (lx, ly, lz), self.sdf_grid,
            bounds_error=False, fill_value=1.0
        )

    def sdf(self, pts):
        pts = np.atleast_2d(pts)
        return self.interp(pts)

    def on_surface(self, pts):
        sdf_vals = self.sdf(pts)
        return np.abs(sdf_vals) < self.surface_thresh, sdf_vals

    def surface_normal(self, pt):
        eps = self.voxel_size/2
        pt = np.array(pt, dtype=np.float64)
        grad = np.array([
            self.sdf(pt + [eps,0,0]) - self.sdf(pt - [eps,0,0]),
            self.sdf(pt + [0,eps,0]) - self.sdf(pt - [0,eps,0]),
            self.sdf(pt + [0,0,eps]) - self.sdf(pt - [0,0,eps]),
        ]).flatten() / (2 * eps)
        norm = np.linalg.norm(grad)
        return grad / norm if norm > 0 else grad

    def sample_surface(self, num_points):
        points, face_idx = trimesh.sample.sample_surface(self.mesh, num_points)
        normals = self.mesh.face_normals[face_idx]
        return points, normals
    
    def find_zero_crossing_quadratic(self,x1, y1, x2, y2, x3, y3):
        """세 점(좌표+SDF값)으로 이차 보간하여 영점(표면) 위치를 추정한다."""
        v = x2 - x1
        vnorm = np.linalg.norm(v)
        if vnorm < 1e-12:
            return None
        v /= vnorm

        mask = v != 0
        if mask.sum() == 0:
            return None
        t1 = 0.0
        t2 = ((x2 - x1)[mask] / v[mask])[0]
        t3 = ((x3 - x1)[mask] / v[mask])[0]

        X = np.array([[t1**2, t1, 1],
                    [t2**2, t2, 1],
                    [t3**2, t3, 1]])
        try:
            w = np.linalg.solve(X, [y1, y2, y3])
        except np.linalg.LinAlgError:
            return None

        # 실수 양의 근 탐색 (0 ~ t3 범위)
        t_zc = None
        for r in np.roots(w):
            if np.isreal(r) and 0 <= r.real <= t3:
                t_zc = r.real
                break

        if t_zc is None:
            if np.abs(w[0]) < 1e-10:
                return None
            t_zc = -w[1] / (2 * w[0])
            if t_zc < -1.0 or t_zc > t3 + 1.0:
                return None

        return x1 + t_zc * v
    
    def stable_poses(self):
        pose,prob=self.mesh.compute_stable_poses()
        return pose, prob