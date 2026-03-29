# depth_image.py
import logging

import numpy as np
import cv2


class DepthImage:
    def __init__(self, depth_data=None,visualize=None):
        self._data = depth_data.astype(np.float32)
        self._data[np.isnan(self._data)] = 0.0
        logging.debug(f'이미지 데이터 크기: {self._data.shape}')
        self._sobel_x = None
        self._sobel_y = None
        self.visualize=visualize

    def _compute_sobel(self):
        if self._sobel_x is None:
            # # depth=0 영역을 주변값으로 채워서 가짜 gradient 방지
            # mask = (self._data == 0).astype(np.uint8)
            # self._data = cv2.inpaint(self._data, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
            if self.visualize:
                cv2.imshow('raw',self._data)
                cv2.waitKey(0)
            self._sobel_x = cv2.Sobel(self._data, cv2.CV_32F, 1, 0, ksize=3)
            self._sobel_y = cv2.Sobel(self._data, cv2.CV_32F, 0, 1, ksize=3)

    def gradient_threshold(self, threshold=0.015,visualize=False):
        if self.visualize is None:
            self.visualize=visualize
        self._compute_sobel()
        grad = np.hypot(self._sobel_x, self._sobel_y)
        logging.debug(f'grad값 {grad}')
        # edge = np.copy(self._data)
        edge=np.ones_like(self._data)
        edge[grad > threshold] = 0.0
        if self.visualize:
            cv2.imshow('edge',edge)
            cv2.waitKey(0)
        return edge  # ndarray 리턴 (DepthImage로 감쌀 필요 없음)

    def surface_normals(self, edge_image):
        """edge_image: gradient_threshold() 리턴 (H,W) ndarray"""
        self._compute_sobel()
        normals = np.zeros((*edge_image.shape, 2), dtype=np.float32)
        ys, xs = np.where(edge_image == 0)
        raw = np.stack([self._sobel_y[ys, xs], self._sobel_x[ys, xs]], axis=1)
        mag = np.linalg.norm(raw, axis=1, keepdims=True)
        mag = np.where(mag == 0, 1, mag)
        normals[ys, xs] = raw / mag

        if self.visualize:
            canvas = cv2.normalize(self._data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            step = max(1, len(ys) // 200)
            arrow_len = 15
            ##for i in range(0, len(ys), step):
            for i in range(len(ys)):
                y, x = ys[i], xs[i]
                ny, nx = normals[y, x]
                pt1 = (int(x), int(y))
                pt2 = (int(x + nx * arrow_len), int(y + ny * arrow_len))
                cv2.arrowedLine(canvas, pt1, pt2, (0, 255, 0), 1, tipLength=0.3)
                cv2.circle(canvas, pt1, 1, (0, 0, 255), -1)
            cv2.imshow("Surface Normals", canvas)
            cv2.waitKey(0)

        return normals

    def resize(self, scale):
        resized = cv2.resize(self._data, None, fx=scale, fy=scale)
        return DepthImage(resized)


if __name__ == "__main__":
    import Minsoo_net.online.online_camera 
    logging.basicConfig(level=logging.DEBUG)

    # 깊이 이미지 로드
    camera=Minsoo_net.online.online_camera.RealSenseCamera()
    camera.update_frames()
    raw=camera.get_depth_image()._data
    di = DepthImage(raw,visualize=True)

    # edge 추출
    edge = di.gradient_threshold(threshold=0.015)
    ys, xs = np.where(edge == 0)
    print(f"edge 픽셀 수: {len(ys)}")

    # normal 계산
    normals = di.surface_normals(edge)