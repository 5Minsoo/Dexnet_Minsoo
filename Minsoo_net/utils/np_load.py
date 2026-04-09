import cv2
import numpy as np

image = cv2.imread('/home/minsoo/Dexnet_Minsoo/Minsoo_net/test/saved_data/depth_raw_4.png', cv2.IMREAD_UNCHANGED)

mask = (image == 0).astype(np.uint8)  # 단일 채널 uint8

# 만약 16bit depth면 inpaint용으로 8bit 변환 필요
if image.dtype == np.uint16:
    image_8bit = (image / 256).astype(np.uint8)
else:
    image_8bit = image

result = cv2.inpaint(image_8bit, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

cv2.imshow('original', image_8bit)
cv2.imshow('inpainted', result)
cv2.waitKey(0)
cv2.destroyAllWindows()