import numpy as np
import cv2

class OnlineAntipodalSampler:
    def __init__(self, gripper_width_m, K, max_grasps=100):
        self.gripper_width_m = gripper_width_m  # 그리퍼 최대 너비 (미터 단위)
        self.K = np.array(K)                    # 3x3 카메라 내부 파라미터 행렬
        # 역행렬을 미리 계산해 둡니다. (3D Unprojection 연산 속도 최적화용)
        self.K_inv = np.linalg.inv(self.K)
        self.max_grasps = max_grasps

    def sample_grasps(self, depth_image, roi=None):
        """
        객체 마스크 없이 깊이 이미지와 ROI를 기반으로 4-DOF 파지 후보군을 반환합니다.
        roi: [x_min, y_min, x_max, y_max] 형태의 리스트 또는 튜플
        """
        sampled_grasps = []
        
        # 1. 깊이 이미지에서 직접 엣지(경계) 추출
        # 깊이 이미지를 0~255 스케일의 8bit 이미지로 정규화하여 Canny 적용
        # (현업에서는 바닥 평면 등 불필요한 배경 깊이를 먼저 0으로 날려버리기도 합니다)
        depth_norm = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        edges = cv2.Canny(depth_norm, 10, 50) 
        
        # ROI가 주어졌다면, 관심 영역 밖의 엣지를 모두 지워버림(Masking)
        if roi is not None:
            x_min, y_min, x_max, y_max = roi
            roi_mask = np.zeros_like(edges)
            roi_mask[y_min:y_max, x_min:x_max] = 255
            edges = cv2.bitwise_and(edges, roi_mask)
            
        edge_pixels = np.argwhere(edges > 0) # [y, x] 형태의 배열
        
        if len(edge_pixels) < 2:
            print('edge가 없습니다')
            return sampled_grasps

        # 2. 이미지 그래디언트 (표면 법선) 계산
        grad_x = cv2.Sobel(depth_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_image, cv2.CV_64F, 0, 1, ksize=3)

        # 3. 무작위 대척점(Antipodal) 쌍 샘플링
        max_iterations = self.max_grasps * 20 
        
        for _ in range(max_iterations):
            if len(sampled_grasps) >= self.max_grasps:
                break
                
            # 무작위 두 점 선택
            idx1, idx2 = np.random.choice(len(edge_pixels), 2, replace=False)
            p1, p2 = edge_pixels[idx1], edge_pixels[idx2] # p = [v(y), u(x)]
            
            z1 = depth_image[p1[0], p1[1]]
            z2 = depth_image[p2[0], p2[1]]
            
            # 센서 노이즈로 인한 유효하지 않은 깊이값(0 또는 음수) 방어 로직
            if z1 <= 0 or z2 <= 0:
                continue
                
            # -----------------------------------------------------------
            # 핵심 변경: Intrinsic 행렬 K를 이용한 정확한 3D 좌표 복원
            # -----------------------------------------------------------
            # 동차 좌표계(Homogeneous coordinates) 구성: [u, v, 1]
            uv1 = np.array([p1[1], p1[0], 1.0])
            uv2 = np.array([p2[1], p2[0], 1.0])
            
            # 카메라 좌표계 상의 3D 포인트 계산 (역투영)
            # 논문의 파지 중심점 계산 공식 c = (1/d)K^-1(p_x, p_y, 1)^T 의 원리와 동일합니다[cite: 609].
            pt1_3d = z1 * self.K_inv.dot(uv1)
            pt2_3d = z2 * self.K_inv.dot(uv2)
            
            # 정확한 유클리디안 물리적 거리(그리퍼 요구 너비) 계산
            phys_dist = np.linalg.norm(pt1_3d - pt2_3d)
            
            # 조건 A: 물리적 거리가 그리퍼 너비를 초과하면 기각
            if phys_dist*0.001 > self.gripper_width_m:
                # print(f'그리퍼 너비 ({self.gripper_width_m})> {phys_dist}')
                continue 
            
            # -----------------------------------------------------------
            # 표면 법선 벡터 및 대척점(Antipodal) 검사
            # -----------------------------------------------------------
            n1 = np.array([grad_x[p1[0], p1[1]], grad_y[p1[0], p1[1]]])
            n2 = np.array([grad_x[p2[0], p2[1]], grad_y[p2[0], p2[1]]])
            
            n1 = n1 / (np.linalg.norm(n1) + 1e-6)
            n2 = n2 / (np.linalg.norm(n2) + 1e-6)
            
            if np.dot(n1, n2) < -0.7: 
                # 파지 중심점
                center_v = (p1[0] + p2[0]) / 2.0
                center_u = (p1[1] + p2[1]) / 2.0
                center_depth = depth_image[int(center_v), int(center_u)]
                
                # 그리퍼 회전 각도
                dv = p2[0] - p1[0]
                du = p2[1] - p1[1]
                theta = np.arctan2(dv, du)
                
                sampled_grasps=[center_u,  center_v,     theta,    center_depth ]
                
        return sampled_grasps

# --- 사용 예시 ---
# K_matrix = [[525.0, 0.0, 319.5], 
#             [0.0, 525.0, 239.5], 
#             [0.0, 0.0, 1.0]]
# sampler = OnlineAntipodalSampler(gripper_width_m=0.05, K=K_matrix, max_grasps=100)
# ROI 설정 (예: x 100~500, y 100~400 박스 안에서만 검색)
# candidates = sampler.sample_grasps(depth_image_array, roi=[100, 100, 500, 400])