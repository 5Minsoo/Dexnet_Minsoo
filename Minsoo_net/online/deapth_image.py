import numpy as np
import cv2
import online_camera

class DepthImage:
    def __init__(self,depth_data):
        self._data=depth_data.astype(np.float32)
        self._data[np.isnan(self._data)] = 0.0

    def gradient_threshold(self, threshold=15):
        sobel_x = cv2.Sobel(self._data, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(self._data, cv2.CV_32F, 0, 1, ksize=3)
        
        grad = np.hypot(sobel_x, sobel_y)
        
        mask = grad > threshold
        edge_pixels = np.copy(self._data)
        edge_pixels[mask] = 0.0
        return edge_pixels

    def resize(self,scale):
        depth=np.copy(self._data)
        return cv2.resize(depth,None,fx=scale,fy=scale)
    
    def surface_normals(self,edge_pixels):
        """
        모든 edge의 normal vector를 구함 Return (N,2)
        """
        sobel_x = cv2.Sobel(self._data, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(self._data, cv2.CV_32F, 0, 1, ksize=3)

        normals = np.zeros([edge_pixels.shape[0], 2])
        for i, px in enumerate(edge_pixels):
            dx=sobel_x[px[0],px[1]]
            dy=sobel_y[px[0],px[1]]
            normal_vec=np.array([dy,dx])
            if np.linalg.norm(normal_vec) == 0:
                normal_vec = np.array([1, 0])
            normal_vec = normal_vec / np.linalg.norm(normal_vec)
            normals[i, :] = normal_vec
        return normals



def nothing(x):
    pass
if __name__ == "__main__":
    camera = online_camera.RealSenseCamera()
    
    # 윈도우 생성 및 슬라이더 등록
    cv2.namedWindow('Depth Control')
    # 'Threshold'라는 이름의 슬라이더 생성 (0~200 범위, 초기값 15)
    cv2.createTrackbar('Threshold', 'Depth Control', 15, 200, nothing)

    try:
        while True:
            camera.update_frames()
            raw_depth = camera.get_depth_image()
            if raw_depth is None:
                continue

            # 1. 슬라이더에서 현재 설정된 threshold 값 읽어오기
            # (정수만 반환되므로 소수점 조절이 필요하면 나누기 10 등을 활용)
            current_th = cv2.getTrackbarPos('Threshold', 'Depth Control')
            print(current_th)
            # 2. 필터링 적용
            depth_obj = DepthImage(raw_depth)
            filtered_depth = depth_obj.gradient_threshold(current_th)

            # 3. 시각화 처리 (0~255 범위로 스케일링)
            # 리얼센스 mm 데이터를 보기 좋게 정규화
            display_depth = cv2.normalize(filtered_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # 엣지가 날아간 부분(0)을 더 잘 보이게 하기 위해 컬러맵 적용
            color_depth = cv2.applyColorMap(display_depth, cv2.COLORMAP_JET)

            # 결과 출력
            cv2.imshow('Depth Control', color_depth)

            # 'q' 누르면 종료, 실시간을 위해 waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()