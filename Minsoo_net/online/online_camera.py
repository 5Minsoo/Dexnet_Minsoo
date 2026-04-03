import math,re

import pyrealsense2 as rs
import numpy as np
import cv2

from Minsoo_net.online.depth_image import DepthImage

def parse_intrinsic_string_to_K(intrinsics):
    """
    RealSense 등의 출력 문자열을 파싱하여 3x3 Intrinsic Matrix (K)로 변환합니다.
    """
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx  # Principal Point X
    cy = intrinsics.ppy  # Principal Point Y

    # 3x3 Numpy 행렬 구성
    K = np.array([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0]
    ])
    
    return K
class RealSenseCamera:
    def __init__(self, width=848, height=480, fps=30):
        # 1. 파이프라인 및 설정 초기화
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.width = width
        self.height = height
        self.align=rs.align(rs.stream.color)
        # 스트림 활성화
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # 2. 스트리밍 시작 및 센서 설정
        self.profile = self.pipeline.start(self.config)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        
        # High Accuracy 프리셋 적용 (4)

        self.depth_sensor.set_option(rs.option.visual_preset, 4)

        # Depth 스케일 (미터 변환용)
        self.depth_scale = self.depth_sensor.get_depth_scale()
        # [중요] 오직 Depth 센서 기준의 Intrinsic (내부 파라미터) 가져오기
        depth_stream = self.profile.get_stream(rs.stream.depth)
        color_stream = self.profile.get_stream(rs.stream.color)
        self.intrinsic_parameter=parse_intrinsic_string_to_K(color_stream.as_video_stream_profile().get_intrinsics())

        # 내부 프레임 버퍼 및 변환 행렬
        self.depth_frame = None
        self.color_frame = None
        self.T_wc = None
        self.T_cw = None
        
        # 포인트 클라우드 계산기 (내장 함수 사용용)
        self.pc = rs.pointcloud()

    def update_frames(self):
        """새로운 프레임 세트를 받아오고 내부 버퍼를 업데이트합니다. (Align 없음)"""
        frames = self.pipeline.wait_for_frames()
        frames=self.align.process(frames)
        # 순수 raw 프레임 사용
        self.depth_frame = frames.get_depth_frame()
        self.color_frame = frames.get_color_frame()
        
        if not self.depth_frame or not self.color_frame:
            return False
        return True

    # 1. Depth Image Return
    def get_depth_image(self):
        """Depth 이미지를 numpy 배열(uint16)로 반환합니다."""
        if not self.depth_frame:
            return None
        return DepthImage(np.asanyarray(self.depth_frame.get_data())*self.depth_scale)

    # 2. Color Image Return (디버깅용)
    def get_color_image(self):
        """Color 이미지를 numpy 배열(uint8)로 반환합니다."""
        if not self.color_frame:
            return None
        return np.asanyarray(self.color_frame.get_data())


    # 6. Debug용 컬러맵 띄우는 함수
    def show_debug_colormap(self):
        """디버깅용: Color 이미지와 Depth 컬러맵을 나란히 출력합니다."""
        color_image = self.get_color_image()
        depth_image = self.get_depth_image()._data
        
        if color_image is None or depth_image is None:
            return

        # Depth 가시화 처리 (0.03은 환경에 따라 조절)
        depth_normalized = cv2.convertScaleAbs(depth_image, alpha=0.03)
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        images = np.hstack((color_image, depth_colormap))
        cv2.imshow('RealSense Debug (Left: Color, Right: Depth)', images)
        cv2.waitKey(1)

    def inside_box_image(self,lower_value= np.array([90, 100, 100]),upper_value= np.array([110, 255, 255]),area_threshold=40000):
        depth=self.get_depth_image()
        color=self.get_color_image()
        if depth is not None:
            hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, lower_value, upper_value)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            color_result = color.copy()
            depth_result = np.zeros_like(depth)
            for i, cnt in enumerate(contours):
                if cv2.contourArea(cnt) < area_threshold:
                    continue
                contour_mask = np.zeros(color.shape[:2], dtype=np.uint8)
                cv2.drawContours(contour_mask, [cnt], -1, 255, cv2.FILLED)

                color_result[contour_mask == 0] = (0, 0, 0)
                depth_result[contour_mask == 255] = depth[contour_mask == 255]
            
            return color_result,depth_result
        
    @staticmethod
    def crop_and_rotate_batch(image, grasps, crop_size=(200, 200)):
        """
        image: 원본 Depth 이미지 (2D 배열)
        grasps: Sampler에서 반환된 (N, 4) 배열 [u, v, theta, depth]
        반환값: N개의 크롭된 이미지 배열 (N, crop_size[1], crop_size[0])
        """
        cropped_images = []
        
        for grasp in grasps:
            u, v, theta, _ = grasp
            cx, cy = float(u), float(v)
            
            # 1. 각도 계산 (Sampler의 theta는 라디안이므로 도(Degree)로 변환)
            angle_deg = math.degrees(theta)
            rotation_angle = angle_deg + 90 
            
            # 2. 중심점 기준 회전 행렬 생성
            M = cv2.getRotationMatrix2D((cx, cy), rotation_angle, 1.0)
            
            # 3. [최적화 핵심] 잘라낼 영역의 중심이 결과 이미지의 정중앙에 오도록 평행 이동
            M[0, 2] += (crop_size[0] / 2.0) - cx
            M[1, 2] += (crop_size[1] / 2.0) - cy
            
            # 4. 회전과 크롭을 한 방에 처리
            # 🚨 Depth 이미지는 보간(Interpolation)을 하면 경계선 픽셀 값이 뭉개지므로 
            # 반드시 INTER_NEAREST를 써야 물리적인 깊이값이 훼손되지 않습니다.
            cropped_image = cv2.warpAffine(
                image, 
                M, 
                crop_size, 
                flags=cv2.INTER_NEAREST, 
                borderMode=cv2.BORDER_CONSTANT, 
                borderValue=0
            )
            
            cropped_images.append(cropped_image)
            
        # 파이토치 모델에 넣기 좋게 (N, H, W) 형태의 다차원 넘파이 배열로 묶어서 반환
        return np.array(cropped_images, dtype=image.dtype)

    @staticmethod
    def show_images(color=None, depth=None, window_name="Camera View"):
        frames = []

        if color is not None:
            if len(color.shape) == 2:
                color = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)
            frames.append(color)

        if depth is not None:
            depth_float = depth.astype(np.float32)
            mask = depth_float > 0

            if mask.any():
                d_min, d_max = depth_float[mask].min(), depth_float[mask].max()
                depth_norm = np.zeros_like(depth_float, dtype=np.uint8)
                if d_max - d_min > 0:
                    depth_norm[mask] = ((depth_float[mask] - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                else:
                    depth_norm[mask] = 128
                depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                depth_colormap[~mask] = (0, 0, 0)
            else:
                depth_colormap = np.zeros((*depth.shape, 3), dtype=np.uint8)
            frames.append(depth_colormap)

        if not frames:
            return

        combined = np.hstack(frames) if len(frames) > 1 else frames[0]
        cv2.imshow(window_name, combined)
        cv2.waitKey(1)

    def release(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    camera = RealSenseCamera()
    try:
        while True:
            camera.update_frames()
            depth=camera.get_depth_image()._data
            print(depth.max())
    finally:
        camera.release()