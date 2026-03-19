import pyrealsense2 as rs
import numpy as np
import cv2
import math,re
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
    def __init__(self, width=640, height=480, fps=30):
        # 1. 파이프라인 및 설정 초기화
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.width = width
        self.height = height

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
        self.intrinsic_parameter=parse_intrinsic_string_to_K(depth_stream.as_video_stream_profile().get_intrinsics())

        # 내부 프레임 버퍼 및 변환 행렬
        self.depth_frame = None
        self.color_frame = None
        self.T_wc = None
        self.T_cw = None
        
        # 포인트 클라우드 계산기 (내장 함수 사용용)
        self.pc = rs.pointcloud()

    def set_extrinsic(self, T_wc):
        """
        카메라-월드 간의 Extrinsic 행렬을 클래스 내부에 설정합니다.
        T_wc: Camera to World Extrinsic Matrix (SE3, 4x4 numpy array)
        """
        self.T_wc = T_wc
        self.T_cw = np.linalg.inv(T_wc) # 역행렬 미리 계산 (월드 -> 카메라용)

    def update_frames(self):
        """새로운 프레임 세트를 받아오고 내부 버퍼를 업데이트합니다. (Align 없음)"""
        frames = self.pipeline.wait_for_frames()
        
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
        return np.asanyarray(self.depth_frame.get_data())

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
        depth_image = self.get_depth_image()
        
        if color_image is None or depth_image is None:
            return

        # Depth 가시화 처리 (0.03은 환경에 따라 조절)
        depth_normalized = cv2.convertScaleAbs(depth_image, alpha=0.03)
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        images = np.hstack((color_image, depth_colormap))
        cv2.imshow('RealSense Debug (Left: Color, Right: Depth)', images)
        cv2.waitKey(1)

    # 7. Axis, Center 기준 Crop & Rotate (16-bit Depth 안전 처리 포함)
    @staticmethod
    def crop_and_rotate(image, center, axis, crop_size=(200, 200)):
        """
        center를 중앙으로 하고, 지정된 axis가 수직 방향이 되도록 회전 및 크롭합니다.
        Depth 이미지(uint16) 입력 시에도 안전하게 작동하도록 보강됨.
        """
        cx, cy = center
        dx, dy = axis
        
        # 각도 계산 후 수직 지향 보정 (+90도)
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        rotation_angle = angle_deg + 90 

        # OpenCV의 getRectSubPix는 16-bit uint를 직접 지원하지 않으므로 float32로 변환 후 처리
        is_uint16 = image.dtype == np.uint16
        if is_uint16:
            image = image.astype(np.float32)

        # 1) 회전 매트릭스 생성 및 회전 적용
        M = cv2.getRotationMatrix2D((cx, cy), rotation_angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        # 2) 크롭
        cropped_image = cv2.getRectSubPix(rotated_image, crop_size, (cx, cy))
        
        # 다시 uint16으로 복원 (Depth 값 유지)
        if is_uint16:
            cropped_image = np.round(cropped_image).astype(np.uint16)
            
        return cropped_image

    def release(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()


# ==========================================
# 실행 예시 (Main)
# ==========================================
if __name__ == "__main__":
    camera = RealSenseCamera()
    
    # 예시용 Extrinsic (로봇 베이스 기준 카메라의 위치/자세 등)
    # 실제 환경에서는 캘리브레이션된 4x4 변환 행렬을 넣어주세요.
    T_wc_dummy = np.eye(4)
    camera.set_extrinsic(T_wc_dummy) # 클래스에 Extrinsic 세팅

    try:
        while True:
            if not camera.update_frames():
                continue
            
            # 1. Depth 이미지 집중 활용
            depth_img = camera.get_depth_image()

            # 2. 디버깅 화면 출력
            camera.show_debug_colormap()
            
            # 3. 좌표 변환 테스트
            # 화면 중앙 픽셀의 월드 좌표 구하기
            center_u, center_v = 320, 240
            world_pt = camera.pixel_to_world(center_u, center_v)
            if world_pt is not None:
                # 출력량이 너무 많아지므로 주석 처리
                # print(f"중앙 픽셀 월드 좌표: {world_pt}")
                pass

            # 전체 포인트 클라우드 월드 좌표로 한 방에 얻기
            all_world_pts = camera.get_world_pointcloud()
            
            # 4. Depth 이미지 Crop & Rotate 테스트 (16-bit 데이터 그대로 유지됨)
            axis_vector = (1, 1) 
            cropped_depth = camera.crop_and_rotate(depth_img, (center_u, center_v), axis_vector, crop_size=(150, 150))
            
            # 잘라낸 Depth 이미지 시각화용 (확인용으로 컬러맵 입힘)
            cropped_depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(cropped_depth, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow('Cropped & Rotated Depth', cropped_depth_vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.release()