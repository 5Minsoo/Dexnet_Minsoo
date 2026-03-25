import cv2
import torch
import time
import numpy as np
from online_camera import RealSenseCamera
from online_sampler import OnlineAntipodalSampler
from Minsoo_net.model.model import DexNet2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DexNet2.load('/home/minsoo/Dexnet_Minsoo/output/20260324_15-07/best.pt')
model.to(device)
model.eval()
use_visual = True
camera = RealSenseCamera()
K = camera.intrinsic_parameter
sampler = OnlineAntipodalSampler(0.05, K)

# ──────────────────────────────────────────────
# 샘플 누적 설정
# ──────────────────────────────────────────────
ACCUMULATE_SEC = 1.0      # 샘플을 모으는 시간 (초)
sample_buffer = []          # 누적된 샘플 리스트
accumulate_start = None     # 누적 시작 시각
is_accumulating = False     # 현재 누적 중인지
best_grasp = None           # 최종 베스트 파지 (추론 결과)
best_score = None
top_grasps = None           # 상위 20개 파지 [(grasp, score), ...]

# ──────────────────────────────────────────────
# ROI 관련 상태 변수
# ──────────────────────────────────────────────
roi = None
drawing = False
ix, iy = -1, -1
temp_roi = None
img_width = None

def _to_depth_x(x):
    if img_width is not None and x >= img_width:
        return x - img_width
    return x

def mouse_callback(event, x, y, flags, param):
    global roi, drawing, ix, iy, temp_roi
    dx = _to_depth_x(x)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = dx, y
        temp_roi = None
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_roi = (min(ix, dx), min(iy, y),
                        abs(dx - ix), abs(y - iy))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, dx), min(iy, y)
        w, h = abs(dx - ix), abs(y - iy)
        if w > 10 and h > 10:
            roi = (x1, y1, w, h)
            print(f"[ROI 설정] x={x1}, y={y1}, w={w}, h={h}")
        else:
            print("[ROI] 영역이 너무 작아 무시됨")
            temp_roi = None

WINDOW_NAME = "Dex-Net Real-time"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

print("=" * 50)
print("  조작법")
print(f"  - SPACE 키     : 샘플 누적 시작 ({ACCUMULATE_SEC}초)")
print("  - 마우스 드래그 : ROI 설정")
print("  - 'r' 키       : ROI 초기화")
print("  - 'q' 키       : 종료")
print("=" * 50)

while True:
    try:
        camera.update_frames()
        color, depth = camera.inside_box_image()
        depth=depth*camera.depth_scale
        img_width = depth.shape[1]

        # ── ROI 적용 ──────────────────────────────
        if roi is not None:
            rx, ry, rw, rh = roi
            rx = max(0, min(rx, depth.shape[1] - 1))
            ry = max(0, min(ry, depth.shape[0] - 1))
            rw = min(rw, depth.shape[1] - rx)
            rh = min(rh, depth.shape[0] - ry)
            depth_roi = depth[ry:ry+rh, rx:rx+rw]
        else:
            rx, ry = 0, 0
            depth_roi = depth

        # ── 누적 모드일 때: 샘플 수집 ─────────────
        if is_accumulating:
            elapsed = time.time() - accumulate_start
            samples = sampler.sample_grasps(depth_roi)

            if samples is not None and len(samples) > 0:
                # ROI 오프셋 보정
                samples[:, 0] += rx
                samples[:, 1] += ry
                sample_buffer.append(samples)

            # 시간 종료 → 모델 추론
            if elapsed >= ACCUMULATE_SEC:
                is_accumulating = False

                if len(sample_buffer) > 0:
                    all_samples = np.vstack(sample_buffer)
                    before = len(all_samples)
                    all_samples = np.unique(all_samples, axis=0)
                    total = len(all_samples)
                    print(f"[추론] {elapsed:.1f}초 동안 {before}개 샘플 누적 → 중복 제거 후 {total}개 → 모델 추론 시작")

                    cropped = RealSenseCamera.crop_and_rotate_batch(
                        depth, all_samples, crop_size=(96, 96)
                    )
                    cropped = np.array([
                        cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
                        for img in cropped
                    ])
                    cropped_input = np.expand_dims(cropped, axis=-1)
                    poses_input = all_samples[:, 3].reshape(-1, 1)
                    success_probs = model.predict_success(cropped_input, poses_input)

                    top_k = 20
                    top_indices = np.argsort(success_probs.flatten())[::-1][:top_k]
                    print(f"\n{'='*50}")
                    print(f"  Top {top_k} / {len(success_probs)}")
                    print(f"  {'rank':>4}  |  {'prob':>8}  |  {'u':>6}  {'v':>6}  {'θ':>6}  {'z':>6}")
                    print(f"  {'-'*4}  |  {'-'*8}  |  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
                    for rank, i in enumerate(top_indices):
                        p = success_probs.flatten()[i]
                        s = all_samples[i]
                        bar = '█' * int(p * 20)
                        print(f"  {rank+1:4d}  |  {p*100:7.2f}%  |  {s[0]:6.0f}  {s[1]:6.0f}  {s[2]:6.2f}  {s[3]:6.3f}  {bar}")
                    print(f"{'='*50}\n")

                    best_idx = np.argmax(success_probs)
                    best_grasp = all_samples[best_idx]
                    best_score = success_probs[best_idx]
                    top_grasps = [(all_samples[i], success_probs.flatten()[i]) for i in top_indices]
                    print(f"[결과] Best score: {best_score*100:.1f}%, "
                          f"위치: u={best_grasp[0]:.0f}, v={best_grasp[1]:.0f}, "
                          f"depth={best_grasp[3]:.1f}")
                else:
                    print("[추론] 누적된 샘플이 없습니다.")
                    best_grasp = None
                    best_score = None
                    top_grasps = None

                sample_buffer.clear()

        # ── 시각화 ────────────────────────────────
        if use_visual:
            depth_norm = cv2.normalize(
                depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            depth_color = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)

            # 누적 중 상태 표시
            if is_accumulating:
                elapsed = time.time() - accumulate_start
                remaining = max(0, ACCUMULATE_SEC - elapsed)
                count = sum(len(s) for s in sample_buffer)
                cv2.putText(
                    depth_color,
                    f"Collecting... {remaining:.1f}s left ({count} samples)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                )
                # 누적 중 프로그레스 바
                bar_w = int((elapsed / ACCUMULATE_SEC) * (depth.shape[1] - 20))
                cv2.rectangle(depth_color, (10, 50), (10 + bar_w, 65), (0, 255, 255), -1)
                cv2.rectangle(depth_color, (10, 50), (depth.shape[1] - 10, 65), (0, 255, 255), 1)

            # 베스트 파지 시각화 (추론 완료 후 유지)
            elif best_grasp is not None and best_score is not None:
                # 상위 20개 작게 표시 (노란색)
                if top_grasps is not None:
                    for grasp, score in top_grasps[1:]:  # best 제외
                        u_t, v_t, theta_t, z_t = grasp
                        ct = (int(u_t), int(v_t))
                        ll = 12
                        dx_t = int(ll * np.cos(theta_t))
                        dy_t = int(ll * np.sin(theta_t))
                        cv2.line(depth_color,
                                 (ct[0] - dx_t, ct[1] - dy_t),
                                 (ct[0] + dx_t, ct[1] + dy_t),
                                 (0, 255, 255), 1)
                        cv2.circle(depth_color, ct, 2, (0, 200, 200), -1)
                        cv2.putText(depth_color, f"{score*100:.0f}%",
                                    (ct[0] + 5, ct[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

                # Best grasp 크게 표시 (초록색)
                u, v, theta, z = best_grasp
                center = (int(u), int(v))
                line_len = 30
                ddx = int(line_len * np.cos(theta))
                ddy = int(line_len * np.sin(theta))
                cv2.line(
                    depth_color,
                    (center[0] - ddx, center[1] - ddy),
                    (center[0] + ddx, center[1] + ddy),
                    (0, 255, 0), 3,
                )
                cv2.circle(depth_color, center, 6, (0, 0, 255), -1)
                cv2.putText(
                    depth_color,
                    f"Best: {best_score*100:.1f}%",
                    (center[0] + 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
                cv2.putText(
                    depth_color,
                    f"Best Grasp: {best_score*100:.1f}%, Depth: {best_grasp[3]:.3f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )
            else:
                cv2.putText(
                    depth_color, "Press SPACE to start sampling", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2,
                )

            # ROI 사각형
            if roi is not None:
                cv2.rectangle(depth_color, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 2)
                cv2.rectangle(color, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 2)

            # 드래그 중 임시 사각형
            if temp_roi is not None:
                tx, ty, tw, th = temp_roi
                cv2.rectangle(depth_color, (tx, ty), (tx + tw, ty + th), (0, 255, 255), 1)

            result = np.hstack((color, depth_color))
            cv2.imshow(WINDOW_NAME, result)

        # ── 키 입력 처리 ─────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("종료합니다.")
            break
        elif key == ord('r'):
            roi = None
            temp_roi = None
            best_grasp = None
            best_score = None
            top_grasps = None
            print("[ROI 초기화] 전체 영역으로 복귀")
        elif key == ord(' '):
            if not is_accumulating:
                is_accumulating = True
                accumulate_start = time.time()
                sample_buffer.clear()
                best_grasp = None
                best_score = None
                top_grasps = None
                print(f"[누적 시작] {ACCUMULATE_SEC}초 동안 샘플 수집 중...")

    except Exception as e:
        print(f"루프 중단 (에러 또는 카메라 해제): {e}")
        break

camera.release()
cv2.destroyAllWindows()