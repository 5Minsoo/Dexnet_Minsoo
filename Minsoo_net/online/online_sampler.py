import numpy as np
import cv2
import deapth_image
from scipy.spatial.distance import pdist, squareform
def force_closure(p1s, p2s, n1s, n2s, mu):
    v = p2s - p1s  # (N, 2)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    alpha = np.arctan(mu)
    dot_1 = np.clip(np.sum(n1s * (-v), axis=1), -1, 1)
    dot_2 = np.clip(np.sum(n2s * v, axis=1), -1, 1)
    return (np.arccos(dot_1) < alpha) & (np.arccos(dot_2) < alpha)  # (N,) bool


class OnlineAntipodalSampler:
    def __init__(self, gripper_width_m, K, max_grasps=100):
        self.gripper_width_m = gripper_width_m  
        self.K = np.array(K)                    
        self.K_inv = np.linalg.inv(self.K)
        self.max_grasps = max_grasps
        self.grad_threshold=0.1

    def sample_grasps(self, depth_image):
        """
        깊이 이미지를 기반으로 N개의 4-DOF 파지 후보군을 배치로 반환합니다.
        반환 형태: (N, 4) 크기의 NumPy 배열 [u(x), v(y), theta, depth]
        """
        sampled_grasps = []
        
        edge_pixels=depth_image.gradient_threshold()
        ys,xs=np.where(edge_pixels==0)
        pixels=np.stack([ys,xs],axis=1)
        dists=pdist(pixels)
        
        min_depth = np.min(edge_pixels)
        max_depth = np.max(edge_pixels)
        
        edge_normals=depth_image.surface_normals(edge_pixels)

        max_iterations = self.max_grasps * 20 
        
        # 💡 [최적화 파라미터] 초점거리 및 탐색 반경(m) 설정
        fx = self.K[0, 0] 
        max_reach_m = self.gripper_width_m * 1.5 
        
        for _ in range(max_iterations):
            if len(sampled_grasps) >= self.max_grasps:
                break
                
            # -----------------------------------------------------------
            # [Step 1] 점 1(p1) 무작위 선택
            # -----------------------------------------------------------
            idx1 = np.random.choice(len(edge_pixels))
            p1 = edge_pixels[idx1] 
            z1 = depth_image[p1[0], p1[1]]
            if z1 <= 0:
                continue
                
            # -----------------------------------------------------------
            # [Step 2] p1의 깊이(z1)에서 그리퍼 1.5배 너비가 몇 픽셀인지 계산
            # (z1 단위가 mm이므로, max_reach_m을 mm로 변환(*1000)하여 계산) -> 수정됨
            # -----------------------------------------------------------
            pixel_radius = (max_reach_m * fx) / z1
            # -----------------------------------------------------------
            # [Step 3] p1 주변 pixel_radius 반경 이내의 모서리 점들만 필터링
            # -----------------------------------------------------------
            diff = edge_pixels - p1
            dist_sq = diff[:, 0]**2 + diff[:, 1]**2 
            radius_sq = pixel_radius**2
            
            # 거리가 0보다 크고(자기 자신 제외), 반경 이내인 점들의 인덱스 찾기
            valid_indices = np.where((dist_sq > 0) & (dist_sq <= radius_sq))[0]
            
            # 탐색 반경 내에 짝꿍 점이 없으면 이번 루프는 포기
            if len(valid_indices) == 0:
                continue
                
            # -----------------------------------------------------------
            # [Step 4] 반경 안쪽의 유효한 점들 중에서 점 2(p2) 무작위 선택
            # -----------------------------------------------------------
            idx2 = np.random.choice(valid_indices)
            p2 = edge_pixels[idx2]
            z2 = depth_image[p2[0], p2[1]]
            
            if z2 <= 0:
                continue
                
            # -----------------------------------------------------------
            # [Step 5] 3D 좌표 복원 및 거리, 표면 법선 검사
            # -----------------------------------------------------------
            uv1 = np.array([p1[1], p1[0], 1.0])
            uv2 = np.array([p2[1], p2[0], 1.0])
            
            pt1_3d = z1 * self.K_inv.dot(uv1)
            pt2_3d = z2 * self.K_inv.dot(uv2)
            
            phys_dist = np.linalg.norm(pt1_3d - pt2_3d)
            
            if phys_dist * 0.001 > self.gripper_width_m:
                continue 
            
            n1 = np.array([grad_x[p1[0], p1[1]], grad_y[p1[0], p1[1]]])
            n2 = np.array([grad_x[p2[0], p2[1]], grad_y[p2[0], p2[1]]])
            
            n1 = n1 / (np.linalg.norm(n1) + 1e-6)
            n2 = n2 / (np.linalg.norm(n2) + 1e-6)
            
            if np.dot(n1, n2) < -0.7:  ###################################
                center_v = (p1[0] + p2[0]) / 2.0
                center_u = (p1[1] + p2[1]) / 2.0
                center_depth = depth_image[int(center_v), int(center_u)]
                if center_depth <0.005:
                    continue
                
                dv = p2[0] - p1[0]
                du = p2[1] - p1[1]
                theta = np.arctan2(dv, du)
                offset=[0.0,0.01,0.02,0.03]*1000
                for offset in offset:
                    sampled_grasps.append([center_u, center_v, theta, center_depth+offset])
                
        return np.array(sampled_grasps, dtype=np.float32)