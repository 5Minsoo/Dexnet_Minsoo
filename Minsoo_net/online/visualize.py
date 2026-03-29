import numpy as np
import cv2



class GraspVisualizer2D:
    """깊이 이미지 위에 파지 후보(center, axis, theta)를 cv2로 시각화하는 클래스."""

    def __init__(self, line_length=30, line_thickness=2, center_radius=4,
                 grasp_color=(0, 255, 0), center_color=(0, 0, 255), alpha=0.6):
        self.line_length = line_length       # 축 선 반길이 (픽셀)
        self.line_thickness = line_thickness
        self.center_radius = center_radius
        self.grasp_color = grasp_color       # BGR
        self.center_color = center_color     # BGR
        self.alpha = alpha

    def _normalize_image(self, image):
        """깊이 이미지를 3채널 uint8로 변환."""
        if image.dtype != np.uint8:
            img = image.copy().astype(np.float64)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
            img = img.astype(np.uint8)
        else:
            img = image.copy()

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

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
            axes = np.stack([np.sin(thetas), np.cos(thetas)], axis=1)
        elif thetas is None and axes is not None:
            thetas = np.arctan2(axes[:, 0], axes[:, 1])
        elif axes is None and thetas is None:
            has_direction = False          # ← 둘 다 None

        n = len(centers)
        if max_show is not None and max_show < n:
            idx = np.random.choice(n, max_show, replace=False)
            centers = centers[idx]
            if has_direction:
                axes, thetas = axes[idx], thetas[idx]
            n = max_show

        canvas = self._normalize_image(image)
        overlay = canvas.copy()

        half = self.line_length
        jaw = int(half * 0.35)

        for i in range(n):
            cy, cx = int(centers[i, 0]), int(centers[i, 1])

            if has_direction:
                dy, dx = axes[i]
                x0, y0 = int(cx - dx * half), int(cy - dy * half)
                x1, y1 = int(cx + dx * half), int(cy + dy * half)
                cv2.line(overlay, (x0, y0), (x1, y1),
                        self.grasp_color, self.line_thickness)

                nx, ny = -dy, dx
                for ex, ey in [(x0, y0), (x1, y1)]:
                    jx0, jy0 = int(ex - nx * jaw), int(ey - ny * jaw)
                    jx1, jy1 = int(ex + nx * jaw), int(ey + ny * jaw)
                    cv2.line(overlay, (jx0, jy0), (jx1, jy1),
                            self.grasp_color, self.line_thickness)

            cv2.circle(overlay, (cx, cy), self.center_radius,
                    self.center_color, -1)

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
        """
        Parameters
        ----------
        image  : (H, W) ndarray
        grasps : (N, 4) ndarray — [u(x), v(y), theta, depth]
        """
        grasps = np.asarray(grasps)
        grasps = np.atleast_2d(np.asarray(grasps))  # (4,) → (1, 4)
        # u, v 기준 unique
        _, idx = np.unique(grasps[:, :2], axis=0, return_index=True)
        grasps = grasps[idx]

        # 열 분리: u=x, v=y
        centers = np.stack([grasps[:, 1], grasps[:, 0]], axis=1)  # (N, 2) [y, x]
        thetas = grasps[:, 2]

        return self.visualize_2d(image, centers, thetas=thetas,
                                title=title, max_show=max_show, wait=wait)
    
    def visualize_debug(self, image, all_samples, success_probs, top_k=20):
        top_indices = np.argsort(success_probs.flatten())[::-1][:top_k]
        best_idx = top_indices[0]

        # 콘솔 출력 (top_k만)
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

        # 시각화 (전체)
        canvas = self._normalize_image(image)
        overlay = canvas.copy()
        half = self.line_length
        jaw = int(half * 0.35)

        # 전체 grasp (초록, 얇게)
        for i in range(len(all_samples)):
            if i == best_idx:
                continue
            s = all_samples[i]
            u, v, theta = s[0], s[1], s[2]
            cx, cy = int(u), int(v)
            dx, dy = np.cos(theta), np.sin(theta)

            x0, y0 = int(cx - dx * half), int(cy - dy * half)
            x1, y1 = int(cx + dx * half), int(cy + dy * half)
            cv2.line(overlay, (x0, y0), (x1, y1), (0, 255, 0), 1)

            nx, ny = -dy, dx
            for ex, ey in [(x0, y0), (x1, y1)]:
                jx0, jy0 = int(ex - nx * jaw), int(ey - ny * jaw)
                jx1, jy1 = int(ex + nx * jaw), int(ey + ny * jaw)
                cv2.line(overlay, (jx0, jy0), (jx1, jy1), (0, 255, 0), 1)

            cv2.circle(overlay, (cx, cy), 2, (0, 200, 0), -1)

        # Best (빨강, 굵게)
        s = all_samples[best_idx]
        u, v, theta = s[0], s[1], s[2]
        cx, cy = int(u), int(v)
        dx, dy = np.cos(theta), np.sin(theta)
        p_best = success_probs.flatten()[best_idx]

        x0, y0 = int(cx - dx * half), int(cy - dy * half)
        x1, y1 = int(cx + dx * half), int(cy + dy * half)
        cv2.line(overlay, (x0, y0), (x1, y1), (0, 0, 255), 3)

        nx, ny = -dy, dx
        for ex, ey in [(x0, y0), (x1, y1)]:
            jx0, jy0 = int(ex - nx * jaw), int(ey - ny * jaw)
            jx1, jy1 = int(ex + nx * jaw), int(ey + ny * jaw)
            cv2.line(overlay, (jx0, jy0), (jx1, jy1), (0, 0, 255), 3)

        cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(overlay, f"BEST {p_best*100:.1f}%", (cx+8, cy-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.addWeighted(overlay, self.alpha, canvas, 1 - self.alpha, 0, canvas)
        cv2.putText(canvas, f"All {len(all_samples)} Grasps (GREEN) + Best (RED)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Debug Grasps", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return canvas
    
# ── 사용 예시 ──
if __name__ == "__main__":
    H, W = 480, 640
    dummy_depth = np.random.rand(H, W).astype(np.float32)

    N = 20
    centers = np.column_stack([
        np.random.randint(50, H - 50, N),
        np.random.randint(50, W - 50, N),
    ])
    thetas = np.random.uniform(-np.pi, np.pi, N)

    viz = GraspVisualizer2D(line_length=25)
    viz.visualize_2d(dummy_depth, centers, thetas=thetas, title="Demo Grasps")