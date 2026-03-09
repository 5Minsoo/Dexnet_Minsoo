import numpy as np
import pyrender
import trimesh


def visualize_grasps(graspable,grasps,pose, sphere_radius=0.002,gripper=None):
    """
    pyrender로 메쉬 + 그래스프 양끝점을 시각화한다.

    Parameters
    ----------
    graspable : GraspableObject3D
        .mesh 속성이 trimesh 객체
    grasps : list of ParallelJawGrasp
        center, axis, open_width 속성 필요
    sphere_radius : float
        끝점 구의 반지름 (m 단위 기준)
    """
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])

    # ---- 메쉬 ----
    mesh = graspable.mesh.copy()
    scene.add(pyrender.Mesh.from_trimesh(mesh), pose=pose)

    # ---- 그래스프 끝점 ----
    colors = [
        [255, 50, 50, 255],
        [50, 50, 255, 255],
        [50, 200, 50, 255],
        [255, 165, 0, 255],
        [200, 50, 200, 255],
        [0, 200, 200, 255],
        [255, 255, 50, 255],
        [150, 75, 0, 255],
    ]

    for i, g in enumerate(grasps):
        v = g.axis / np.linalg.norm(g.axis)
        half_w = g.open_width / 2.0
        p1 = g.center + half_w * v
        p2 = g.center - half_w * v
        T=g.T_grasp_obj
        axis=trimesh.creation.axis( transform=pose@T,axis_length=0.01,origin_size=0.001)
        scene.add(pyrender.Mesh.from_trimesh(axis,smooth=False), pose=np.eye(4))
        axis_world=trimesh.creation.axis( transform=np.eye(4),axis_length=0.08,origin_size=0.003)
        scene.add(pyrender.Mesh.from_trimesh(axis_world,smooth=False), pose=np.eye(4))
        axis_pose=trimesh.creation.axis( transform=pose,axis_length=0.03,origin_size=0.003)
        scene.add(pyrender.Mesh.from_trimesh(axis_pose,smooth=False), pose=np.eye(4))
        if gripper is not None:
            gripper_mesh = gripper.mesh.copy()
            gripper_mesh.apply_transform(pose@T@np.linalg.inv(gripper.T_grasp_gripper))
            scene.add(pyrender.Mesh.from_trimesh(gripper_mesh,smooth=False), pose=np.eye(4))
        color = colors[i % len(colors)]
        sm = trimesh.creation.uv_sphere(radius=sphere_radius)
        sm.visual.vertex_colors = color

        for p in [p1, p2,g.center]:
            T = np.eye(4)
            T[:3, 3] = p
            scene.add(pyrender.Mesh.from_trimesh(sm), pose=pose@T)

    # ---- 조명 + 카메라 ----
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0)
    scene.add(light)

    pyrender.Viewer(scene, use_raymond_lighting=True, window_title='Grasps')


