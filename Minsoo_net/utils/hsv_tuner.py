from Minsoo_net.online import online_sampler
from Minsoo_net.online import online_camera
import cv2
import numpy as np

# 그리퍼 폭 0.05m(5cm)로 가정 (0.5는 너무 커서 샘플링이 안 될 수 있음)
camera = online_camera.RealSenseCamera()
sampler = online_sampler.OnlineAntipodalSampler(0.05, camera.intrinsic_parameter)

def nothing(x):
    pass

window_name = 'HSV_Tuner'
cv2.namedWindow(window_name)

# 트랙바 초기값 설정
cv2.createTrackbar('Low H', window_name, 20, 179, nothing)
cv2.createTrackbar('Low S', window_name, 100, 255, nothing)
cv2.createTrackbar('Low V', window_name, 100, 255, nothing)
cv2.createTrackbar('High H', window_name, 40, 179, nothing)
cv2.createTrackbar('High S', window_name, 255, 255, nothing)
cv2.createTrackbar('High V', window_name, 255, 255, nothing)

try:
    while True:
        if not camera.update_frames():
            continue
            
        color_rgb = camera.get_color_image()
        # 시각화 및 텍스트 출력을 위해 BGR로 변환
        display_img = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2HSV)

        # 트랙바 값 읽기
        l_h = cv2.getTrackbarPos('Low H', window_name)
        l_s = cv2.getTrackbarPos('Low S', window_name)
        l_v = cv2.getTrackbarPos('Low V', window_name)
        h_h = cv2.getTrackbarPos('High H', window_name)
        h_s = cv2.getTrackbarPos('High S', window_name)
        h_v = cv2.getTrackbarPos('High V', window_name)

        lower = np.array([l_h, l_s, l_v])
        upper = np.array([h_h, h_s, h_v])

        # 마스크 생성 및 필터링
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(display_img, display_img, mask=mask)

        # --- 영상 위에 실시간 수치 표시 (Text Overlay) ---
        lower_text = f"Lower: H={l_h} S={l_s} V={l_v}"
        upper_text = f"Upper: H={h_h} S={h_s} V={h_v}"
        
        # 검은색 배경을 살짝 넣어 글씨가 잘 보이게 처리 (선택사항)
        cv2.rectangle(result, (5, 5), (350, 70), (0, 0, 0), -1) 
        
        # 텍스트 그리기 (이미지, 내용, 좌표, 폰트, 크기, 색상, 두께)
        cv2.putText(result, lower_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result, upper_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # ----------------------------------------------

        cv2.imshow(window_name, result)
        cv2.imshow('Mask', mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"\n[최종 결정값]\nlower = np.array([{l_h}, {l_s}, {l_v}])")
            print(f"upper = np.array([{h_h}, {h_s}, {h_v}])")
            break

finally:
    cv2.destroyAllWindows()