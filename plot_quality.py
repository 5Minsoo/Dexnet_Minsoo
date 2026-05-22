import cv2, math
import numpy as np
from matplotlib import pyplot as plt
import torch

from Minsoo_net.model.model import DexNet2

def crop_image(image, height, num, crop_size=96, output_size=32, depth_offset=0.03):
    h, w = image.shape[:2]
    du=int(w/num)
    v=int(h*height)
    u=int(du/2)
    images=[]
    depth=[]

    for i in range(num):
        center=[u,v]
        cropped = cv2.getRectSubPix(image, (crop_size, crop_size), center )
        cropped=cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_AREA)
        images.append(cropped)

        d=image[v, u] + depth_offset
        depth.append(d)

        u += du

    depth=np.array(depth).reshape(-1,1)
    batch = np.stack(images, axis=0)        # (N, H, W)
    batch = np.expand_dims(batch, axis=-1)  # (N, H, W, 1)
    return batch, depth

def plot_quality_map(image, height, num, model):
    images,depth=crop_image(image, height,num)
    score=model.predict_success(images, depth)
    h, w = image.shape[:2]
    du = int(w / num)
    v = int(h * height)
    # crop_image와 동일한 u 위치 재현: u = du/2 + i*du
    us = np.array([du / 2.0 + i * du for i in range(num)])

    fig, (ax_img, ax_score) = plt.subplots(
        2, 1, figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    # --- 위: 이미지 + crop 라인 표시 ---
    ax_img.imshow(image, cmap="gray")
    ax_img.axhline(v, color="black", lw=1, ls="-")          # crop 높이선
    # ax_img.scatter(us, [v] * num, c=score, cmap="jet",       # 위치별 score 색
    #                s=40, edgecolors="k", zorder=3)
    best = int(np.argmax(score))
    # ax_img.scatter(us[best], v, c="lime", s=120, marker="*", # 최고 점수 표시
    #                edgecolors="k", zorder=4)
    ax_img.set_title("Grasp quality map")
    ax_img.set_xlim(0, w)
    ax_img.axis("off")
    ax_img.imshow(image, cmap="gray", aspect="auto")
    # --- 아래: score 그래프 ---
    ax_score.plot(us, score, "-", color="black")
    # ax_score.scatter(us[best], score[best], c="lime", s=120, # 최고점 강조
    #                  marker="*", edgecolors="k", zorder=3)
    ax_score.set_xlim(0, w)
    ax_score.set_ylim(0, 1)
    ax_score.set_xlabel("u (pixel)")
    ax_score.set_ylabel("success score")
    ax_score.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return score, us

if __name__ == "__main__":
   image_path= '/home/minsoo/Dexnet_Minsoo/Minsoo_net/test/saved_data2/raw_depth_data_1.npz'
   model_path='/home/minsoo/Dexnet_Minsoo/output/04-30_17-00_grasp_dataset_ABC_th0.002/best.pt'
   image=np.load(image_path)
   image= image['depth']
   device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model=DexNet2.load(model_path)
   model.to(device)
   plot_quality_map(image,0.5,100,model)