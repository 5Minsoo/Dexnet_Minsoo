import numpy as np
import cv2
import math


class GraspVisualizer2D:
    """깊이 이미지 위에 파지 후보(center, axis, theta)를 cv2로 시각화하는 클래스."""

    def __init__(self, line_length=50, line_thickness=2, arrow_thickness=2,
                 jaw_ratio=0.3, arrow_ratio=0.5, arrow_head=0.3, cross_size=10,
                 grasp_color=(0, 255, 0), best_color=(0, 0, 255), alpha=0.6):
        self.line_length = line_length
        self.line_thickness = line_thickness
        self.arrow_thickness = arrow_thickness
        self.jaw_ratio = jaw_ratio
        self.arrow_ratio = arrow_ratio
        self.arrow_head = arrow_head
        self.cross_size = cross_size
        self.grasp_color = grasp_color
        self.best_color = best_color
        self.alpha = alpha

    def _normalize_image(self, image):
        if image.dtype != np.uint8:
            img = image.copy().astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
            img = img.astype(np.uint8)
        else:
            img = image.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def _draw_grasp_marks(self, vis, cx, cy, ux, uy, half,
                          color, thickness, arrow_thickness, cross_size):
        """DexNet 스타일 grasp 마커: ㅣ + → → ㅣ + 중심 십자"""
        jaw = int(half * self.jaw_ratio)
        arrow_len = int(half * self.arrow_ratio)

        # axis에 수직인 단위벡터
        nx, ny = -uy, ux

        # 양 끝점 (jaw 중심)
        x0, y0 = int(cx - ux * half), int(cy - uy * half)
        x1, y1 = int(cx + ux * half), int(cy + uy * half)

        # 1) 양 끝 jaw (axis에 수직인 짧은 선)
        for ex, ey in [(x0, y0), (x1, y1)]:
            jx0, jy0 = int(ex - nx * jaw), int(ey - ny * jaw)
            jx1, jy1 = int(ex + nx * jaw), int(ey + ny * jaw)
            cv2.line(vis, (jx0, jy0), (jx1, jy1), color, thickness)

        # 2) 양 끝 바깥 → jaw 안쪽 화살표
        ax0 = int(x0 - ux * arrow_len)
        ay0 = int(y0 - uy * arrow_len)
        cv2.arrowedLine(vis, (ax0, ay0), (x0, y0),
                        color, arrow_thickness, tipLength=self.arrow_head)
        ax1 = int(x1 + ux * arrow_len)
        ay1 = int(y1 + uy * arrow_len)
        cv2.arrowedLine(vis, (ax1, ay1), (x1, y1),
                        color, arrow_thickness, tipLength=self.arrow_head)

        # 3) 중심 십자
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

    def visualize_2d(self, image, centers, axes=None, thetas=None,
                     title="Grasp Candidates", max_show=None, wait=True):
        centers = np.asarray(centers)
        if axes is not None:
            axes = np.asarray(axes)
        if thetas is not None:
            thetas = np.asarray(thetas)

        has_direction = True
        if axes is None and thetas is not None:
            axes = np.stack([np.sin(thetas), np.cos(thetas)], axis=1)
        elif thetas is None and axes is not None:
            thetas = np.arctan2(axes[:, 0], axes[:, 1])
        elif axes is None and thetas is None:
            has_direction = False

        n = len(centers)
        if max_show is not None and max_show < n:
            idx = np.random.choice(n, max_show, replace=False)
            centers = centers[idx]
            if has_direction:
                axes, thetas = axes[idx], thetas[idx]
            n = max_show

        canvas = self._normalize_image(image)
        overlay = canvas.copy()

        for i in range(n):
            cy, cx = int(centers[i, 0]), int(centers[i, 1])
            if has_direction:
                dy, dx = axes[i]
                mag = math.hypot(dx, dy)
                if mag < 1e-8:
                    continue
                ux, uy = dx / mag, dy / mag
                self._draw_grasp_marks(
                    overlay, cx, cy, ux, uy, half=self.line_length,
                    color=self.grasp_color,
                    thickness=self.line_thickness,
                    arrow_thickness=self.arrow_thickness,
                    cross_size=self.cross_size,
                )
            else:
                cv2.circle(overlay, (cx, cy), 4, self.grasp_color, -1)

        cv2.addWeighted(overlay, self.alpha, canvas, 1 - self.alpha, 0, canvas)

        cv2.putText(canvas, f"{title}  (n={n})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow(title, canvas)
        if wait:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return canvas

    def visualize_from_grasps(self, image, grasps, title="Grasp Candidates",
                              max_show=None, wait=True):
        grasps = np.atleast_2d(np.asarray(grasps))
        _, idx = np.unique(grasps[:, :2], axis=0, return_index=True)
        grasps = grasps[idx]

        centers = np.stack([grasps[:, 1], grasps[:, 0]], axis=1)  # [y, x]
        thetas = grasps[:, 2]

        return self.visualize_2d(image, centers, thetas=thetas,
                                 title=title, max_show=max_show, wait=wait)

    def visualize_debug(self, image, all_samples, success_probs,
                        top_k=20, max_show=None):
        """
        all_samples: (N, 4+) [u, v, theta, z, ...]
        success_probs: (N,) 또는 (N, 1)
        top_k: print 출력용 top-k
        max_show: 화면에 그릴 grasp 최대 개수 (None이면 전체)
        """
        success_probs = success_probs.flatten()
        top_indices = np.argsort(success_probs)[::-1][:top_k]
        best_idx = top_indices[0]

        # ── print 로그 ──
        print(f"\n{'='*60}")
        print(f"  Top {top_k} / {len(success_probs)}")
        print(f"  {'rank':>4}  |  {'prob':>8}  |  {'u':>6}  {'v':>6}  {'θ':>6}  {'z':>6}")
        print(f"  {'-'*4}  |  {'-'*8}  |  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
        for rank, i in enumerate(top_indices):
            p = success_probs[i]
            s = all_samples[i]
            bar = '█' * int(p * 20)
            marker = " ★ BEST" if i == best_idx else ""
            print(f"  {rank+1:4d}  |  {p*100:7.2f}%  |  {s[0]:6.0f}  {s[1]:6.0f}  {s[2]:6.2f}  {s[3]:6.3f}  {bar}{marker}")
        print(f"{'='*60}\n")

        # ── 그릴 grasp 인덱스 결정 ──
        # best_idx는 항상 포함, 나머지는 max_show-1개 랜덤 샘플
        n = len(all_samples)
        if max_show is not None and max_show < n:
            others = np.array([i for i in range(n) if i != best_idx])
            chosen = np.random.choice(others, size=max_show - 1, replace=False)
            draw_indices = np.concatenate([chosen, [best_idx]])
        else:
            draw_indices = np.arange(n)

        canvas = self._normalize_image(image)
        overlay = canvas.copy()

        # ── 일반 grasp (초록) ──
        for i in draw_indices:
            if i == best_idx:
                continue
            s = all_samples[i]
            u, v, theta = s[0], s[1], s[2]
            ux, uy = math.cos(theta), math.sin(theta)
            self._draw_grasp_marks(
                overlay, int(u), int(v), ux, uy, half=self.line_length,
                color=self.grasp_color,
                thickness=self.line_thickness,
                arrow_thickness=self.arrow_thickness,
                cross_size=self.cross_size,
            )

        # ── Best (빨강, 굵게) ──
        s = all_samples[best_idx]
        u, v, theta = s[0], s[1], s[2]
        ux, uy = math.cos(theta), math.sin(theta)
        p_best = success_probs[best_idx]

        self._draw_grasp_marks(
            overlay, int(u), int(v), ux, uy, half=self.line_length,
            color=self.best_color,
            thickness=self.line_thickness + 1,
            arrow_thickness=self.arrow_thickness + 1,
            cross_size=self.cross_size,
        )

        cv2.putText(overlay, f"BEST {p_best*100:.1f}%",
                    (int(u) + 15, int(v) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.best_color, 2)

        cv2.addWeighted(overlay, self.alpha, canvas, 1 - self.alpha, 0, canvas)

        shown = len(draw_indices)
        cv2.putText(canvas,
                    f"Showing {shown}/{n} grasps + Best (DexNet style)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Debug Grasps", canvas)
        cv2.waitKey(0)
        return canvas

    def visualize_cropped_debug(
            self,
            cropped_images: np.ndarray,
            max_show: int = None,
            upscale: int = 4,
            wait_time: int = 0
        ) -> None:
            """
            크롭된 32x32 이미지 배열(또는 리스트)을 입력받아 가로 방향 마커를 그리고 하나씩 순차적으로 띄웁니다.
            
            :param cropped_images: (N, H, W) 또는 (N, H, W, C) 배열
            :param max_show: 보여줄 최대 이미지 개수
            :param upscale: 시각화를 위한 이미지 확대 배율 (32x32는 너무 작아 4배 기본 설정)
            :param wait_time: cv2.waitKey 대기 시간 (0이면 무한 대기, 1 이상이면 ms 단위로 자동 넘어감)
            """
            if isinstance(cropped_images, list):
                cropped_images = np.array(cropped_images)

            n = len(cropped_images)
            if max_show is not None and max_show < n:
                idx = np.random.choice(n, max_show, replace=False)
                cropped_images = cropped_images[idx]
                n = max_show

            for i, img in enumerate(cropped_images):
                # 1. 정규화 및 컬러 이미지 변환
                canvas = self._normalize_image(img)
                h, w = canvas.shape[:2]

                # 2. 마커를 선명하게 그리기 위해 그리기 전 업스케일링 (INTER_NEAREST로 픽셀 깨짐 방지)
                if upscale > 1:
                    canvas = cv2.resize(canvas, (w * upscale, h * upscale), interpolation=cv2.INTER_NEAREST)
                    h, w = canvas.shape[:2]

                # 3. 중앙 및 가로축 설정
                cx, cy = w // 2, h // 2
                half = int((w / 2) * 0.6) # 너비의 80%

                # 마커 그리기
                self._draw_grasp_marks(
                    canvas, cx, cy, ux=1.0, uy=0.0, half=half,
                    color=self.grasp_color,
                    thickness=1 if upscale < 3 else 2,
                    arrow_thickness=1 if upscale < 3 else 2,
                    cross_size=max(2, int(w * 0.1))
                )

                # 4. 이미지 띄우기
                cv2.imshow("Cropped Debug", canvas)
                
                # 키 입력 대기 (아무 키나 누르면 다음 이미지, q나 ESC 누르면 종료)
                key = cv2.waitKey(wait_time) & 0xFF
                if key == ord('q') or key == 27:
                    print("시각화를 중단합니다.")
                    break

            cv2.destroyAllWindows()

# ── 사용 예시 ──
if __name__ == "__main__":
    H, W = 480, 640
    object_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(object_mask, (W//2, H//2), 100, 1, -1)
    dist_transform = cv2.distanceTransform(object_mask, cv2.DIST_L2, 5)
    dummy_depth = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)

    N = 200
    centers = np.column_stack([
        np.random.randint(H//2 - 80, H//2 + 80, N),
        np.random.randint(W//2 - 80, W//2 + 80, N),
    ])
    thetas = np.random.uniform(-np.pi, np.pi, N)

    viz = GraspVisualizer2D(line_length=30, jaw_ratio=0.5, arrow_ratio=0.8, cross_size=6)

    # 200개 중 30개만 보여주기
    viz.visualize_2d(dummy_depth, centers, thetas=thetas,
                     title="Grasp Candidates", max_show=3)

    all_samples = np.column_stack([
        centers[:, 1], centers[:, 0], thetas,
        np.random.uniform(0.1, 0.5, N),
    ])
    success_probs = np.random.rand(N)

    # debug: 200개 중 best 포함 20개만 그리기
    viz.visualize_debug(dummy_depth, all_samples, success_probs,
                        top_k=10, max_show=3)
    # === 기존 테스트 코드 아래에 이어서 추가 ===
    
    # 가상의 32x32 크롭 이미지 50개 생성 (테스트용)
    dummy_crops = np.random.randint(0, 255, (50, 32, 32), dtype=np.uint8)
    
    # 최대 24개만 8열(가로 8개) 그리드로 보여주기 (4배 확대 -> 128x128 캔버스)
    viz.visualize_cropped_debug(dummy_crops, max_show=24, upscale=4)