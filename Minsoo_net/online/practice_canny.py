from online_camera import RealSenseCamera
import cv2
if __name__ == '__main__':
    camera = RealSenseCamera()
    try:
        while True:
            camera.update_frames()
            color, depth = camera.inside_box_image()
            depth=depth/camera.depth_scale
            print(depth.max())
            depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth=cv2.Canny(depth_norm,10,50)
            cv2.imshow('depth',depth)
            cv2.waitKey(0)

    finally:
        camera.release()