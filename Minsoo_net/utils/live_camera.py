import pyrealsense2 as rs
import numpy as np
import cv2

# 1. 파이프라인 및 설정 초기화
pipeline = rs.pipeline()
config = rs.config()

# RGB와 Depth 스트림 활성화 (해상도와 FPS 설정)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 2. 스트리밍 시작
selsection=pipeline.start(config)
depth_sensor=selsection.get_device().first_depth_sensor()
preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
print(f"사용 가능한 프리셋 범위: {preset_range.min} ~ {preset_range.max}")

# High Accuracy 프리셋 적용
depth_sensor.set_option(rs.option.visual_preset, 4) 

print("프리셋 적용 완료: High Accuracy")
try:
    while True:
        # 3. 프레임 세트 대기 (새로운 데이터가 올 때까지 대기)
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # 4. 프레임을 넘파이 배열로 변환
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_normalized = cv2.convertScaleAbs(depth_image, alpha=0.3)
        color_image = np.asanyarray(color_frame.get_data())

        # 시각화를 위해 Depth 맵에 색상 입히기
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # 이미지 출력
        images = np.hstack((color_image, depth_colormap))
        cv2.imshow('RealSense', images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 5. 파이프라인 종료
    pipeline.stop()
    cv2.destroyAllWindows()