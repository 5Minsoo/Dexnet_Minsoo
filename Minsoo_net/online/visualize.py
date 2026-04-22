import numpy as np
import cv2

class GraspVisualizer2D:
    """깊이 이미지 위에 파지 후보(center, axis, theta)를 cv2로 시각화하는 클래스."""

    def __init__(self, line_length=50, line_thickness=2, center_radius=4,
                 arrow_length=20, # 화살표 꼬리 길이
                 grasp_color=(0, 255, 0), center_color=(0, 0, 255), alpha=0.6):
        self.line_length = line_length       # 축 선 반길이 (픽셀)
        self.line_thickness = line_thickness
        self.center_radius = center_radius
        self.arrow_length = arrow_length     
        self.grasp_color = grasp_color       # BGR
        self.center_color = center_color     # BGR
        self.alpha = alpha

    def _normalize_image(self, image):
        """깊이 이미지를 3채널 uint8로 변환."""
        if image.dtype != np.uint8:
            img = image.copy().astype(np.float32) 
            img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
            img = img.astype(np.uint8)
        else:
            img = image.copy()

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def _draw_grasp_element(self, img, cx, cy, dx, dy, color, thickness, is_best=False, draw_arrows=True):
        """단일 파지 형태(가로선, 세로 조, 바깥->안쪽 화살표)를 렌더링하는 내부 메서드"""
        half = self.line_length
        jaw = int(half * 0.35)
        arrow_len = self.arrow_length

        # 1. 세로 조(Jaw)의 중심점(p1, p2)
        x0, y0 = int(cx - dx * half), int(cy - dy * half)
        x1, y1 = int(cx + dx * half), int(cy + dy * half)

        # 2. 파지 중심 가로선 연결
        cv2.line(img, (x0, y0), (x1, y1), color, thickness)

        # 3. 바깥에서 안쪽(조 중심)을 향하는 화살표 (--> <--) - 요청에 따라 조건부 그리기
        if draw_arrows:
            x0_out, y0_out = int(x0 - dx * arrow_len), int(y0 - dy * arrow_len)
            x1_out, y1_out = int(x1 + dx * arrow_len), int(y1 + dy * arrow_len)
            # tipLength를 늘려 화살표 머리를 더 크게 만듦 (예: 0.3 -> 0.45)
            cv2.arrowedLine(img, (x0_out, y0_out), (x0, y0), color, thickness, tipLength=0.45)
            cv2.arrowedLine(img, (x1_out, y1_out), (x1, y1), color, thickness, tipLength=0.45)

        # 4. 세로 조(Jaw) 그리기 (수직 방향) - ㅣ-ㅣ 형태의 세로선
        nx, ny = -dy, dx
        for ex, ey in [(x0, y0), (x1, y1)]:
            jx0, jy0 = int(ex - nx * jaw), int(ey - ny * jaw)
            jx1, jy1 = int(ex + nx * jaw), int(ey + ny * jaw)
            cv2.line(img, (jx0, jy0), (jx1, jy1), color, thickness)

        # 5. 중심점 표시
        radius = self.center_radius + 2 if is_best else self.center_radius
        if is_best:
            cv2.circle(img, (int(cx), int(cy)), radius, color, -1)


    def visualize_2d(self, image, centers, axes=None, thetas=None,
                    title="Grasp Candidates", max_show=None, wait=True):
        centers = np.asarray(centers)
        if axes is not None:
            axes = np.asarray(axes)
        if thetas is not None:
            thetas = np.asarray(thetas)

        # axes ↔ thetas 상호 보완
        has_direction = True
        if axes is None and thetas is not None:
            axes = np.stack([np.sin(thetas), np.cos(thetas)], axis=1) # (dy, dx) 형태
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
                # 일반 후보들은 화살표 없이 그리도록 draw_arrows=False 설정
                self._draw_grasp_element(overlay, cx, cy, dx, dy, self.grasp_color, self.line_thickness, draw_arrows=False)
            else:
                cv2.circle(overlay, (cx, cy), self.center_radius, self.center_color, -1)

        cv2.addWeighted(overlay, self.alpha, canvas, 1 - self.alpha, 0, canvas)

        label = f"{title}  (n={n})"
        cv2.putText(canvas, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow(title, canvas)
        if wait:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return canvas

    def visualize_from_grasps(self, image, grasps, title="Grasp Candidates",
                            max_show=None, wait=True):
        grasps = np.asarray(grasps)
        grasps = np.atleast_2d(np.asarray(grasps))
        _, idx = np.unique(grasps[:, :2], axis=0, return_index=True)
        grasps = grasps[idx]

        centers = np.stack([grasps[:, 1], grasps[:, 0]], axis=1)  # [y, x]
        thetas = grasps[:, 2]

        return self.visualize_2d(image, centers, thetas=thetas,
                                title=title, max_show=max_show, wait=wait)
    
    def visualize_debug(self, image, all_samples, success_probs, top_k=20):
        top_indices = np.argsort(success_probs.flatten())[::-1][:top_k]
        best_idx = top_indices[0]

        print(f"\n{'='*60}")
        print(f"  Top {top_k} / {len(success_probs)}")
        print(f"  {'rank':>4}  |  {'prob':>8}  |  {'u':>6}  {'v':>6}  {'θ':>6}  {'z':>6}")
        print(f"  {'-'*4}  |  {'-'*8}  |  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
        for rank, i in enumerate(top_indices):
            p = success_probs.flatten()[i]
            s = all_samples[i]
            bar = '█' * int(p * 20)
            marker = " ★ BEST" if i == best_idx else ""
            print(f"  {rank+1:4d}  |  {p*100:7.2f}%  |  {s[0]:6.0f}  {s[1]:6.0f}  {s[2]:6.2f}  {s[3]:6.3f}  {bar}{marker}")
        print(f"{'='*60}\n")

        canvas = self._normalize_image(image)
        overlay = canvas.copy()

        # 전체 grasp (초록, 얇게)
        for i in range(len(all_samples)):
            if i == best_idx:
                continue
            s = all_samples[i]
            u, v, theta = s[0], s[1], s[2]
            dx, dy = np.cos(theta), np.sin(theta)
            
            # 일반 후보들은 화살표 없이 그리도록 draw_arrows=False 설정
            self._draw_grasp_element(overlay, u, v, dx, dy, (0, 255, 0), 2, draw_arrows=False)

        # Best (빨강, 굵게)
        s = all_samples[best_idx]
        u, v, theta = s[0], s[1], s[2]
        dx, dy = np.cos(theta), np.sin(theta)
        p_best = success_probs.flatten()[best_idx]

        # Best 파지만 화살표를 그리도록 draw_arrows=True 설정 (기본값)
        self._draw_grasp_element(overlay, u, v, dx, dy, (0, 0, 255), 3, is_best=True, draw_arrows=True)

        cv2.putText(overlay, f"BEST {p_best*100:.1f}%", (int(u)+15, int(v)-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.addWeighted(overlay, self.alpha, canvas, 1 - self.alpha, 0, canvas)
        cv2.putText(canvas, f"All {len(all_samples)} Grasps (ㅣ-ㅣ) + Best (--> <--) ", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Debug Grasps", canvas)
        cv2.waitKey(0)
        return canvas
    
# ── 사용 예시 ──
if __name__ == "__main__":
    H, W = 480, 640
    # 더 현실적인 더미 깊이 이미지 생성 (물체 주변으로 그라데이션)
    dummy_depth = np.zeros((H, W), dtype=np.float32)
    object_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(object_mask, (W//2, H//2), 100, 1, -1)
    dist_transform = cv2.distanceTransform(object_mask, cv2.DIST_L2, 5)
    dummy_depth = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)

    N = 20
    centers = np.column_stack([
        np.random.randint(H//2 - 80, H//2 + 80, N), # y
        np.random.randint(W//2 - 80, W//2 + 80, N), # x
    ])
    thetas = np.random.uniform(-np.pi, np.pi, N)

    viz = GraspVisualizer2D(line_length=30, arrow_length=20)
    # 일반 시각화 테스트 - 일반 후보들은 화살표 없이 ㅣ-ㅣ 모양으로 보임
    viz.visualize_2d(dummy_depth, centers, thetas=thetas, title="Grasp Candidates (ㅣ-ㅣ Shape)")

    # 디버그 시각화 테스트를 위한 데이터 생성
    all_samples = np.column_stack([
        centers[:, 1], # u (x)
        centers[:, 0], # v (y)
        thetas,
        np.random.uniform(0.1, 0.5, N) # z (depth)
    ])
    success_probs = np.random.rand(N)

    # 디버그 시각화 테스트 - 일반 후보는 초록색 ㅣ-ㅣ, Best는 빨간색 화살표 모양으로 보임
    viz.visualize_debug(dummy_depth, all_samples, success_probs, top_k=10)