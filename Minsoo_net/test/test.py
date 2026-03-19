from Minsoo_net.online import online_sampler
from Minsoo_net.online import online_camera
import cv2
import numpy as np

# 그리퍼 폭 0.05m(5cm)로 가정 (0.5는 너무 커서 샘플링이 안 될 수 있음)
camera = online_camera.RealSenseCamera()
sampler = online_sampler.OnlineAntipodalSampler(0.05, camera.intrinsic_parameter)

try:
    while True:
        # 1. 프레임 업데이트 (한 번만 호출)
        if not camera.update_frames():
            continue
            
        depth = camera.get_depth_image()
        # 시각화용 8비트 이미지 생성 (안 그러면 imshow에서 제대로 안 보임)
        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET) # 컬러풀하게 보기

        # 2. 파지 샘플링
        samples = sampler.sample_grasps(depth)
        
        # 3. 결과 시각화
        if len(samples) > 0:
            # 딕셔너리 키 'u', 'v'로 접근 
            u, v = int(samples[0]), int(samples[1]) 
            cv2.circle(depth_vis, (u, v), 5, (0, 0, 255), -1) # 빨간 점
            
            # 각도(theta) 표시 (선 그리기) 
            line_len = 15
            dx = int(line_len * np.cos(samples[2]))
            dy = int(line_len * np.sin(samples[2]))
            cv2.line(depth_vis, (u-dx, v-dy), (u+dx, v+dy), (255, 255, 255), 2)

        cv2.imshow('Dex-Net Online Sampler', depth_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()