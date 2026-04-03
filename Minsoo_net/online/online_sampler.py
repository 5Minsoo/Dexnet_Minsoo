import logging

import numpy as np
import cv2
from scipy.spatial import cKDTree


from Minsoo_net.online.depth_image import DepthImage
from Minsoo_net.online.visualize import GraspVisualizer2D
from Minsoo_net.online.online_camera import RealSenseCamera

logger = logging.getLogger(__name__)

def force_closure(p1s, p2s, n1s, n2s, mu):
    v = p2s - p1s  # (N, 2)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    alpha = np.arctan(mu)
    dot_1 = np.clip(np.sum(n1s * (-v), axis=1), -1, 1)
    dot_2 = np.clip(np.sum(n2s * v, axis=1), -1, 1)
    return (np.arccos(dot_1) < alpha) & (np.arccos(dot_2) < alpha)

def camera_coords(depth_image, pixel_points, K_inv):
    """
    depth_image: (H, W)
    pixel_points: (N, 2), 각 행이 (y,x)
    K: (3, 3)
    return: (N, 3)
    """

    ones = np.ones((len(pixel_points), 1))
    # pixel_points는 (y,x) 순서이므로 K_inv에 맞게 (x,y,1)로 변환
    pixels_uv = np.hstack([pixel_points[:, 1:2], pixel_points[:, 0:1], ones])  # (N, 3) [u,v,1]

    depths = depth_image[pixel_points[:, 0].astype(int),
                         pixel_points[:, 1].astype(int)]  # (N,)

    camera_points = (pixels_uv @ K_inv.T) * depths[:, np.newaxis]  # (N, 3)

    return camera_points

viz=GraspVisualizer2D()

class OnlineAntipodalSampler:
    def __init__(self, gripper_width_m,grad_threshold=0.015, K=None, max_grasps=10000, image_margin=0.10, max_edge=100,visualize=False):
        self.gripper_width_m = gripper_width_m  
        if K is None:
            self.K=np.array([[392.23574829  , 0.     ,    324.36325073],
                            [  0.    ,     392.23574829 ,239.42385864],
                            [  0.     ,      0.         ,  1.        ]])
        else: self.K = np.array(K)                    
        self.K_inv = np.linalg.inv(self.K)
        self.max_grasps = max_grasps
        self.grad_threshold=grad_threshold
        self.max_edge=max_edge
        self.image_margin=image_margin
        self.visualize=visualize

    def sample_grasps(self, depth_image,use_visualize=False):
        """
        깊이 이미지를 기반으로 N개의 4-DOF 파지 후보군을 배치로 반환합니다.
        반환 형태: (N, 4) 크기의 NumPy 배열 [u(x), v(y), theta, depth]
        """
        depth_image=depth_image.inpaint_depth()
        edge = depth_image.gradient_threshold(self.grad_threshold)
        
        h, w = edge.shape[:2]
        margin = self.image_margin
        t, b = int(h * margin), int(h * (1 - margin))
        l, r = int(w * margin), int(w * (1 - margin))
        edge[:t, :] = 1.0
        edge[b:, :] = 1.0
        edge[:, :l] = 1.0
        edge[:, r:] = 1.0

        if use_visualize:
            cv2.imshow('edge',edge)
            cv2.waitKey(0)
        logger.debug(f'edge image size: {edge.shape}')
        logger.debug(f'edge size {edge.size}')

        ys, xs = np.where(edge == 0)
        pixels = np.stack([ys, xs], axis=1) # (N,2)
        pixels = pixels.astype(np.int16)
        if len(pixels)  > self.max_edge:
            idx=np.random.choice(len(pixels),self.max_edge,replace=False)
            pixels=pixels[idx]
        
        logger.debug(f'edge pixel shape: {pixels.shape}')

        # depth=0(무효)인 픽셀 제거 — 안 하면 (0,0,0)에 몰려서 거리 필터 무의미
        valid_depths = depth_image._data[pixels[:, 0], pixels[:, 1]] > 0
        pixels = pixels[valid_depths]
        logger.debug(f'valid depth pixels: {pixels.shape}')

        max_reach_m = self.gripper_width_m *1.5
        min_reach_m=self.gripper_width_m*0.3
        point_cloud=camera_coords(depth_image._data,pixels,self.K_inv)
        logger.debug(f'Camera coord max {np.max(point_cloud)}')
        logger.debug(f'전체 가능 pair 개수: {int(len(point_cloud)*(len(point_cloud)-1)/2)}')
        tree = cKDTree(point_cloud)
        pair_set = tree.query_pairs(r=max_reach_m, output_type='ndarray')  # (M, 2)
        logger.debug(f'실제 max pair 개수: {pair_set.shape}')
        
        dists = np.linalg.norm(point_cloud[pair_set[:, 0]] - point_cloud[pair_set[:, 1]], axis=1)
        mask = dists >= min_reach_m
        pairs = pair_set[mask]
        pairs = np.array(pairs,dtype=np.intp)
        logger.debug(f'실제 min pair 개수: {pairs.shape}')

        # if use_visualize:
        #     midpoints = (pixels[pairs[:, 0]] + pixels[pairs[:, 1]]) / 2.0
        #     viz.visualize_2d(edge, midpoints,title='candidate centers (distance filtered)')

        edge_normals=depth_image.surface_normals(edge)

        p0=pixels[pairs[:,0]]
        p1=pixels[pairs[:,1]]
        n0 = edge_normals[p0[:, 0], p0[:, 1]]  # y0들, x0들로 한번에
        n1 = edge_normals[p1[:, 0], p1[:, 1]]


        ############ 0필터링 추가 #########################
        valid = (np.linalg.norm(n0, axis=1) > 0) & (np.linalg.norm(n1, axis=1) > 0)
        pairs = pairs[valid]
        p0, p1 = p0[valid], p1[valid]
        n0, n1 = n0[valid], n1[valid]
        ############ 0필터링 추가 #########################

        force_closure_mask=force_closure(p0,p1,n0,n1,0.8)
        pairs=pairs[force_closure_mask]
        logger.debug(f'force closure pairs: {pairs.shape}')

        p0 = pixels[pairs[:, 0]]
        p1 = pixels[pairs[:, 1]]
        n0 = n0[force_closure_mask]
        n1 = n1[force_closure_mask]
        centers = (p0 + p1) // 2
        # ================= 디버그 시각화 (p0, p1, 중심점, 연결선, 법선 벡터) =================
        if use_visualize:
            # 깊이 이미지를 0~255 범위로 정규화한 뒤, 색상을 입힐 수 있게 BGR로 변환
            disp_img = cv2.normalize(depth_image._data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            disp_img = cv2.cvtColor(disp_img, cv2.COLOR_GRAY2BGR)

            for pt0, pt1, c, norm0, norm1 in zip(p0, p1, centers, n0, n1):
                # Numpy는 (y, x) 순서, OpenCV는 (x, y) 순서이므로 뒤집어 할당
                x0, y0 = int(pt0[1]), int(pt0[0])
                x1, y1 = int(pt1[1]), int(pt1[0])
                cx, cy = int(c[1]), int(c[0])

                # 1. p0 ~ p1 연결선 (초록색)
                cv2.line(disp_img, (x0, y0), (x1, y1), (0, 255, 0), 1)

                # 2. 중심점 (빨간색)
                cv2.circle(disp_img, (cx, cy), 2, (0, 0, 255), -1)

                # 3. 파지점 p0, p1 (파란색)
                cv2.circle(disp_img, (x0, y0), 2, (255, 0, 0), -1)
                cv2.circle(disp_img, (x1, y1), 2, (255, 0, 0), -1)

                # 4. 법선 벡터 n0, n1 방향 (노란색 화살표)
                # 방향을 눈으로 쉽게 확인하기 위해 화살표 길이를 15픽셀로 스케일업
                scale = 15
                nx0, ny0 = int(x0 + norm0[1] * scale), int(y0 + norm0[0] * scale)
                nx1, ny1 = int(x1 + norm1[1] * scale), int(y1 + norm1[0] * scale)

                cv2.arrowedLine(disp_img, (x0, y0), (nx0, ny0), (0, 255, 255), 1, tipLength=0.3)
                cv2.arrowedLine(disp_img, (x1, y1), (nx1, ny1), (0, 255, 255), 1, tipLength=0.3)

            cv2.imshow("Debug: Force Closure & Normals", disp_img)
            cv2.waitKey(0)
            cv2.destroyWindow("Debug: Force Closure & Normals")

        axes = p1 - p0
        axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
        thetas = np.arctan2(axes[:, 0], axes[:, 1])
            

        depth_mean = cv2.medianBlur(depth_image._data.astype(np.float32), 5)
        depths = depth_mean[centers[:, 0], centers[:, 1]]  # (N,) 끝
        MIN_VALID_DEPTH = 0.15  # 15cm (카메라 스펙상 최소 거리)
        valid_mask = (depths > MIN_VALID_DEPTH)

        centers = centers[valid_mask]
        thetas = thetas[valid_mask]
        depths = depths[valid_mask]
        offsets = np.array([ -0.01, -0.02, -0.03,-0.04])  # 미터 단위 오프셋
        # 각 grasp마다 offset 개수만큼 복제
        N = len(centers)
        K = len(offsets)

        us = np.tile(centers[:, 1], K)          # (N*K,)
        vs = np.tile(centers[:, 0], K)          # (N*K,)
        ts = np.tile(thetas, K)                 # (N*K,)
        ds = np.repeat(offsets, N) + np.tile(depths, K)  # (N*K,)
        logger.debug(f'전체 결과 {us.shape, vs.shape, ts.shape, ds.shape}')
        grasps = np.column_stack([us, vs, ts, ds])  # (N*K, 4)

        # max_grasps 제한
        if len(grasps) > self.max_grasps:
            idx = np.random.choice(len(grasps), self.max_grasps, replace=False)
            grasps = grasps[idx]
        logger.debug(f'최종 결과 {grasps.shape}')
        if use_visualize:
            viz.visualize_from_grasps(depth_image._data, grasps, title="Antipodal Grasps")
        return grasps.astype(np.float32)


if __name__=="__main__":
    logging.basicConfig(level=logging.DEBUG)
    # camera=RealSenseCamera()
    sampler=OnlineAntipodalSampler(gripper_width_m=0.05,grad_threshold=0.015,max_edge=100)
    while True:
        # camera.update_frames()
        # depth=camera.get_depth_image()
        depth=cv2.imread('/home/minsoo/Dexnet_Minsoo/Minsoo_net/test/saved_data/depth_raw_1.png',cv2.IMREAD_GRAYSCALE)
        depth=depth*0.001
        depth=DepthImage(depth)
        grasp=np.float16(sampler.sample_grasps(depth,use_visualize=True))