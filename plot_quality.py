import cv2, math
import numpy as np
from matplotlib import pyplot as plt
import torch
from scipy.signal import find_peaks

from Minsoo_net.model.model import DexNet2

def draw_grasp_marks(
    vis: np.ndarray,
    centers,                      # (N, 2) uv 좌표 배열 [(u, v), ...]
    heights,                      # 길이 N 배열 또는 스칼라(공통 적용)
    thetas=90.0,                   # 길이 N 배열 또는 스칼라(공통 적용), 라디안
    color=(0, 255, 0),
    thickness: int = 2,
    arrow_thickness: int = 2,
    jaw_ratio: float = 0.5,
    arrow_ratio: float = 0.8,
    arrow_head: float = 0.4,
    cross_size: int = 15,
):
    """공통 그리기 루틴: + ㅣ →←ㅣ + (theta만큼 회전). 지정한 uv 좌표들마다 그림."""
    centers = np.asarray(centers).reshape(-1, 2)
    heights = np.broadcast_to(np.asarray(heights), (len(centers),))
    thetas=np.deg2rad(thetas)
    thetas = np.broadcast_to(np.asarray(thetas, dtype=float), (len(centers),)) +np.deg2rad(90)

    for (cu, cv), height, theta in zip(centers, heights, thetas):
        cx, cy = int(cu), int(cv)
        half = int(height) // 2
        jaw = int(half * jaw_ratio)
        arrow_len = int(half * arrow_ratio)

        # theta 방향 단위벡터 (theta=0이면 세로), axis에 수직인 단위벡터
        ux, uy = -np.sin(theta), np.cos(theta)
        nx, ny = -uy, ux

        # 양 끝점 (jaw 중심)
        x0, y0 = int(cx - ux * half), int(cy - uy * half)
        x1, y1 = int(cx + ux * half), int(cy + uy * half)

        # 1) 양 끝 jaw ㅣ ㅣ (axis에 수직)
        for ex, ey in [(x0, y0), (x1, y1)]:
            jx0, jy0 = int(ex - nx * jaw), int(ey - ny * jaw)
            jx1, jy1 = int(ex + nx * jaw), int(ey + ny * jaw)
            cv2.line(vis, (jx0, jy0), (jx1, jy1), color, thickness)

        if arrow_thickness != 0:
            # 2) 양 끝 바깥 → jaw 안쪽 화살표
            ax0 = int(x0 - ux * arrow_len)
            ay0 = int(y0 - uy * arrow_len)
            cv2.arrowedLine(vis, (ax0, ay0), (x0, y0), color, arrow_thickness, tipLength=arrow_head)
            ax1 = int(x1 + ux * arrow_len)
            ay1 = int(y1 + uy * arrow_len)
            cv2.arrowedLine(vis, (ax1, ay1), (x1, y1), color, arrow_thickness, tipLength=arrow_head)

        # 3) 중심 + 마커 (axis 방향과 수직 방향)
        cv2.line(
            vis,
            (int(cx - ux * cross_size), int(cy - uy * cross_size)),
            (int(cx + ux * cross_size), int(cy + uy * cross_size)),
            color, thickness,
        )
        cv2.line(
            vis,
            (int(cx - nx * cross_size), int(cy - ny * cross_size)),
            (int(cx + nx * cross_size), int(cy + ny * cross_size)),
            color, thickness,
        )
    return vis

def crop_image(image, height, num, theta=0.0, crop_size=96, output_size=32, depth_offset=0.03):
    """
    image: 원본 Depth 이미지 (2D 배열)
    height: 샘플링할 행 위치 (0~1, 이미지 높이 비율)
    num: 가로 라인 위에서 뽑을 크롭 개수
    theta: 회전 각도(라디안). 스칼라면 모든 크롭에 동일 적용,
           길이 num 배열이면 크롭별로 다른 각도 적용
    """
    h, w = image.shape[:2]
    du = int(w / num)
    v = int(h * height)
    u = int(du / 2)

    # theta 정규화: 스칼라/배열 모두 길이 num으로 맞춤
    thetas = np.broadcast_to(np.asarray(theta, dtype=np.float32), (num,))

    images = []
    depth = []
    for i in range(num):
        cx, cy = float(u), float(v)

        # 1. theta(라디안) -> 도(Degree) 변환
        angle_deg = math.degrees(thetas[i])

        # 2. 중심점 기준 회전 행렬 생성
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

        # 3. 크롭 중심이 결과 이미지 정중앙에 오도록 평행이동
        M[0, 2] += (crop_size / 2.0) - cx
        M[1, 2] += (crop_size / 2.0) - cy

        # 4. 회전 + 크롭을 한 번에 (crop_size x crop_size)
        cropped = cv2.warpAffine(
            image,
            M,
            (crop_size, crop_size),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # 5. 모델 입력 크기로 리사이즈
        cropped = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_AREA)
        images.append(cropped)

        # 깊이는 회전 중심(u, v)에서 샘플링하므로 회전과 무관
        d = image[v, u] + depth_offset
        depth.append(d)

        u += du

    depth = np.array(depth).reshape(-1, 1)
    batch = np.stack(images, axis=0)        # (N, H, W)
    batch = np.expand_dims(batch, axis=-1)  # (N, H, W, 1)
    return batch, depth

def plot_quality_map(image, height, num, model, theta):
    images,depth=crop_image(image, height,num, theta)
    score=model.predict_success(images, depth)
    peaks, props = find_peaks(score)
    h, w = image.shape[:2]
    du = int(w / num)
    v = int(h * height)
    best_3 = peaks[np.argsort(score[peaks])[::-1][:2]] 
    # crop_image와 동일한 u 위치 재현: u = du/2 + i*du
    us = np.array([du / 2.0 + i * du for i in range(num)])

    u_3 = us[best_3]
    v_3=np.full(best_3.shape,v)
    centers = np.column_stack([u_3, v_3])

    img_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
    cv2.line(vis, (0, v), (w, v), (0, 0, 0), 1)
    vis = draw_grasp_marks(vis, centers, heights=60,arrow_head=0, arrow_ratio=0, arrow_thickness=0, cross_size=10, jaw_ratio=0.3, thetas=theta)    
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    fig, (ax_img, ax_score) = plt.subplots(
        2, 1, figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    # --- 위: 이미지 + crop 라인 표시 ---
    ax_img.imshow(vis, aspect="auto")
    # ax_img.axhline(v, color="black", lw=1, ls="--")          # crop 높이선
    # ax_img.scatter(us, [v] * num, c=score, cmap="jet",       # 위치별 score 색
    #                s=40, edgecolors="k", zorder=3)
    # ax_img.scatter(us[best_3], v_3, c="lime", s=120, marker="*", # 최고 점수 표시
    #                edgecolors="k", zorder=4)
    ax_img.set_title("Grasp quality map")
    ax_img.set_xlim(0, w)
    ax_img.axis("off")
    # ax_img.imshow(image, cmap="gray", aspect="auto")
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
    plt.savefig('my_chart.svg', format='svg')
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
   theta=0.0
   plot_quality_map(image=image,height=0.5,num=100,model=model, theta=theta)