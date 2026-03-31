import sys, logging, os
import cv2
import numpy as np

# 모델 등 불필요한 import 제거, 카메라 모듈만 남김
from Minsoo_net.online.online_camera import RealSenseCamera

sys.path.append('/home/minsoo/Dexnet_Minsoo/Minsoo_net/online')

class DepthSaver:
    def __init__(self):
        self.camera = RealSenseCamera()
        self.save_dir = '/home/minsoo/Dexnet_Minsoo/Minsoo_net/test/saved_data'
        os.makedirs(self.save_dir, exist_ok=True)

    def run(self):
        logging.info("카메라 스트리밍을 시작합니다. 저장하려면 's', 종료하려면 'q'를 누르세요.")
        
        save_count = 0
        while True:
            # 프레임 업데이트 및 Depth 데이터 가져오기
            self.camera.update_frames()
            depth_obj = self.camera.get_depth_image()
            raw_depth = depth_obj._data  # 정규화되지 않은 원본 raw array
            
            # Depth 이미지 띄우기
            # 주의: 16-bit raw 데이터는 값이 작아서 OpenCV로 그냥 띄우면 화면이 거의 까맣게 보일 수 있습니다.
            # (데이터 자체가 날아간 것은 아니니 안심하셔도 됩니다.)
            cv2.imshow('Raw Depth (Press s to save, q to quit)', raw_depth)
            
            key = cv2.waitKey(1) & 0xFF
            
            # 's' 키를 누르면 원본 데이터 그대로 저장
            if key == ord('s'):
                save_count += 1
                
                # 1. numpy 배열(.npz)로 원본 데이터 저장
                npz_path = os.path.join(self.save_dir, f'raw_depth_data_{save_count}.npz')
                np.savez(npz_path, depth=raw_depth)
                
                # 2. 16-bit png 파일로 원본 데이터 저장
                png_path = os.path.join(self.save_dir, f'depth_raw_{save_count}.png')
                depth_uint16 = raw_depth.astype(np.uint16)
                cv2.imwrite(png_path, depth_uint16)
                
                logging.info(f"[{save_count}번째 저장 완료] 정규화 없이 원본이 저장되었습니다: {self.save_dir}")
                
            # 'q' 키를 누르면 종료
            elif key == ord('q'):
                logging.info("프로그램을 종료합니다.")
                break

        cv2.destroyAllWindows()

def main():
    logging.basicConfig(level=logging.INFO)
    saver = DepthSaver()
    saver.run()

if __name__ == '__main__':
    main()