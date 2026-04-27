import numpy as np
import trimesh
import pyrender
import sklearn
# ── 1. 메시 로드 ──
mesh = trimesh.load('/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/object/Frankapanda/1/00001158_0d5aceffdc1f4188ae053ebf_step_000_0048.obj')
def filter_duplicate_stable_poses(transforms, probs, mesh, angle_thresh=0.15):
    # 물체 주축 (PCA)
    pca = sklearn.decomposition.PCA(n_components=3)
    pca.fit(mesh.vertices)
    principal_axes = pca.components_  # (3, 3)

    unique_transforms = []
    unique_probs = []

    for T, p in zip(transforms, probs):
        z_dir = T[:3, 2]

        # z_dir과 각 주축이 이루는 각도 벡터 (3,)
        angles = np.array([
            np.arccos(np.clip(np.abs(z_dir.dot(ax)), 0, 1))
            for ax in principal_axes
        ])

        duplicate = False
        for u_T in unique_transforms:
            u_z_dir = u_T[:3, 2]
            u_angles = np.array([
                np.arccos(np.clip(np.abs(u_z_dir.dot(ax)), 0, 1))
                for ax in principal_axes
            ])
            if np.linalg.norm(angles - u_angles) < angle_thresh:
                duplicate = True
                break

        if not duplicate:
            unique_transforms.append(T)
            unique_probs.append(p)

    return unique_transforms, unique_probs
hull=mesh.convex_hull
trans,probs=hull.compute_stable_poses()
# 1. AABB (월드 좌표계 기준 절대 크기)
aabb_extents = mesh.extents
print(f"AABB 기준 X, Y, Z 크기: {aabb_extents}")
# bounding box의 [min_x, min_y, min_z], [max_x, max_y, max_z] 좌표가 필요하다면:
print(f"AABB 최소/최대 좌표: {mesh.bounds}") 

# 2. OBB (객체에 타이트하게 맞춘 실제 크기)
obb = mesh.bounding_box_oriented
obb_extents = obb.extents
print(f"OBB 기준 가로, 세로, 높이: {obb_extents}")

# (참고) OBB가 어떤 자세로 회전되어 있는지 변환 행렬 확인
print(f"OBB 변환 행렬(Transform):\n{obb.primitive.transform}")
unique_transforms, unique_probs = trans,probs

# ── 4. pyrender 시각화 ──
scene = pyrender.Scene()
spacing = 0.1

# 월드 축
world_axis = trimesh.creation.axis(origin_size=0.002, axis_length=0.4)
scene.add(pyrender.Mesh.from_trimesh(world_axis,smooth=False), pose=np.eye(4))
j=1
scene.add(pyrender.Mesh.from_trimesh(mesh,smooth=False))
for i, (T, p) in enumerate(zip(unique_transforms, unique_probs)):
    if p<max(unique_probs)*0.05:
        continue
    T_offset = T.copy()
    T_offset[2, 3] += j * spacing
    j+=1

    # 물체
    tri_mesh = mesh.copy()
    tri_mesh.visual = trimesh.visual.ColorVisuals(mesh=tri_mesh)
    scene.add(pyrender.Mesh.from_trimesh(tri_mesh,smooth=False), pose=T_offset)

    # 로컬 축
    local_axis = trimesh.creation.axis(transform=T_offset, axis_length=0.08, origin_size=0.001)
    scene.add(pyrender.Mesh.from_trimesh(local_axis,smooth=False))

    print(f'  Pose {i}: prob={p:.4f}, z_dir={T[:3, 2].round(3)}')

# 조명
light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
scene.add(light)

# 뷰어
view = pyrender.Viewer(scene, use_raymond_lighting=True, window_title='Unique Stable Poses')