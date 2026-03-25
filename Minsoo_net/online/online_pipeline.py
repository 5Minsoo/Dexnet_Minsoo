import cv2
import torch
import numpy as np
from online_camera import RealSenseCamera
from online_sampler import OnlineAntipodalSampler
from Minsoo_net.model.model import DexNet2  # 모델 클래스 임포트 필요

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1. 먼저 체크포인트 파일(딕셔너리 형태)을 통째로 불러옵니다.
model = DexNet2.load('/home/minsoo/Dexnet_Minsoo/output/20260324_00-18/best.pt')
model.to(device)
model.eval()

use_visual = True
camera = RealSenseCamera()
K = camera.intrinsic_parameter
sampler = OnlineAntipodalSampler(0.05, K)

while True:
    try:
        camera.update_frames()
        color,depth = camera.inside_box_image()
        
        
        # 2. 파지 후보 추출 (N, 4)
        samples = sampler.sample_grasps(depth)
        
        # 샘플이 1개 이상 존재할 때만 모델 추론 실행
        if len(samples) > 0:
            # 3. 크롭 및 모델 입력용 차원 확장
            # Dex-Net은 32x32 크기를 사용하므로 crop_size 지정
            cropped = RealSenseCamera.crop_and_rotate_batch(depth, samples, crop_size=(32, 32))
            
            # images: (N, 32, 32) -> (N, 32, 32, 1) 채널 추가
            cropped_input = np.expand_dims(cropped, axis=-1) 
            
            # poses: (N, 4) 배열 중 3번째 인덱스인 depth만 가져와서 (N, 1)로 변환
            poses_input = samples[:, 3].reshape(-1, 1)/1000
            
            # 4. 모델 추론 실행 (반환값: N, 2)
            predict_result = model.predict(cropped_input, poses_input)
            
            # 5. 가장 성공 확률이 높은(Best) 파지 포즈 찾기
            success_probs = predict_result[:, 1]
            best_idx = np.argmax(success_probs)
            
            best_grasp = samples[best_idx]
            best_score = success_probs[best_idx]

            # 6. 시각화 (use_visual이 True일 때만)
            if use_visual:
                # 흑백 Depth를 컬러(BGR) 캔버스로 변환
                depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_color = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)
                
                # 🔥 베스트 파지만 초록색(0, 255, 0)으로 굵게 강조해서 그리기
                u, v, theta, z = best_grasp
                center = (int(u), int(v))
                
                line_len = 20
                dx = int(line_len * np.cos(theta))
                dy = int(line_len * np.sin(theta))
                
                cv2.line(depth_color, (center[0]-dx, center[1]-dy), (center[0]+dx, center[1]+dy), (0, 255, 0), 3)
                cv2.circle(depth_color, center, 4, (0, 0, 255), -1) # 중심은 빨간 점
                
                # 성공 확률 텍스트 띄우기
                cv2.putText(depth_color, f"Best Grasp: {best_score*100:.1f}% ,Depth: {best_grasp[3]}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                result=np.hstack((color,depth_color))
                cv2.imshow("Dex-Net Real-time",result)
                cv2.waitKey(0)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("종료합니다.")
                    break

    except Exception as e:
        print(f"루프 중단 (에러 또는 카메라 해제): {e}")
        break

# 자원 해제
camera.release()
cv2.destroyAllWindows()