import numpy as np
import pyrender
import trimesh


def _create_stick_gripper(open_width=0.05, finger_length=0.045,
                          base_height=0.03, tube_radius=0.0015,
                          sections=6, color=[0, 255, 0, 255]):
    """ GraspFactory 시각화 방식. 막대로 grasp 형상화. 현재는 robotiq-hande 전용.
    """
    half_width = open_width / 2.0

    # 가로바: 손가락 base 위치 (z = -finger_length) 에서 y방향으로 ±half_width
    cb2_1 = trimesh.creation.cylinder(
        radius=tube_radius, sections=sections,
        segment=[[0, 0, -finger_length], [0,  half_width, -finger_length]],
    )
    cb2_2 = trimesh.creation.cylinder(
        radius=tube_radius, sections=sections,
        segment=[[0, -half_width, -finger_length], [0, 0, -finger_length]],
    )
    # mount: 가로바에서 더 -z 방향으로 base_height
    cb_1 = trimesh.creation.cylinder(
        radius=tube_radius, sections=sections,
        segment=[[0, 0, -(finger_length + base_height)], [0, 0, -finger_length]],
    )
    # 손가락: base(z=-finger_length) → tip(z=0)
    cf_l = trimesh.creation.cylinder(
        radius=tube_radius, sections=sections,
        segment=[[0,  half_width, -finger_length], [0,  half_width, 0]],
    )
    cf_r = trimesh.creation.cylinder(
        radius=tube_radius, sections=sections,
        segment=[[0, -half_width, -finger_length], [0, -half_width, 0]],
    )

    g = trimesh.util.concatenate([cb_1, cb2_1, cb2_2, cf_l, cf_r])
    g.visual.face_colors = color
    return g

def visualize_grasps(graspable, grasps, pose,
                     finger_length=0.04, tube_radius=0.0015,
                     gripper=None, gripper_type="panda", title=None):
    """
    pyrender로 메쉬 + 그래스프를 시각화한다.

    gripper가 주어지면 실제 그리퍼 메쉬를 씌우고,
    None이면 막대기(stick) 그리퍼로 대체한다.

    Parameters
    ----------
    graspable : GraspableObject3D
        .mesh 속성이 trimesh 객체
    grasps : list of ParallelJawGrasp
        center, axis, open_width, T_grasp_obj 속성 필요
    pose : (4,4) np.ndarray
        object -> world 변환
    finger_length : float
        막대기 그리퍼의 손가락 길이 (m). gripper=None일 때만 사용.
    tube_radius : float
        막대 반지름 (m). gripper=None일 때만 사용.
    gripper : optional
        .mesh, .T_grasp_gripper 속성을 가진 실제 그리퍼.
        주어지면 실제 메쉬를 사용, None이면 막대기 그리퍼 사용.
    gripper_type : str
        "panda" 또는 "robotiq". gripper=None일 때 막대기 형태 결정.
    title : str
        viewer 창 제목
    """
    assert gripper_type in ["panda", "robotiq"]

    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])

    # ---- 메쉬 ----
    mesh = graspable.mesh.copy()
    scene.add(pyrender.Mesh.from_trimesh(mesh), pose=pose)

    # ---- 월드 좌표축 ----
    axis_world = trimesh.creation.axis(
        transform=np.eye(4), axis_length=0.08, origin_size=0.003
    )
    scene.add(pyrender.Mesh.from_trimesh(axis_world, smooth=False), pose=np.eye(4))

    # ---- 오브젝트 pose 좌표축 ----
    axis_pose = trimesh.creation.axis(
        transform=pose, axis_length=0.03, origin_size=0.003
    )
    scene.add(pyrender.Mesh.from_trimesh(axis_pose, smooth=False), pose=np.eye(4))

    # ---- 색상 팔레트 ----
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

    # ---- 그래스프 ----
    for i, g in enumerate(grasps):
        T = g.T_grasp_obj
        color = colors[i % len(colors)]

        # 그래스프 좌표축
        axis_grasp = trimesh.creation.axis(
            transform=pose @ T, axis_length=0.01, origin_size=0.001
        )
        # scene.add(pyrender.Mesh.from_trimesh(axis_grasp, smooth=False),
        #           pose=np.eye(4))

        if gripper is not None:
            # ---- 실제 그리퍼 메쉬 ----
            gripper_mesh = gripper.mesh.copy()
            gripper_mesh.apply_transform(
                pose @ T @ np.linalg.inv(gripper.T_grasp_gripper)
            )
            scene.add(
                pyrender.Mesh.from_trimesh(gripper_mesh, smooth=False),
                pose=np.eye(4),
            )
        else:
            # ---- 막대기 그리퍼 ----
            stick = _create_stick_gripper(
                open_width=g.open_width,
                finger_length=finger_length,
                tube_radius=tube_radius,
                color=color
            )
            stick.apply_transform(pose @ T)
            scene.add(pyrender.Mesh.from_trimesh(stick, smooth=False),
                      pose=np.eye(4))

    # ---- 조명 ----
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0)
    scene.add(light)

    pyrender.Viewer(scene, use_raymond_lighting=True, window_title=title)

def visualize_samples(graspable, grasps, pose,
                     bar_length=0.02, bar_radius=0.0015,
                     title=None):
    """
    pyrender로 메쉬 + 그래스프를 시각화한다.

    Parameters
    ----------
    graspable : GraspableObject3D
        .mesh 속성이 trimesh 객체
    grasps : list of ParallelJawGrasp
        center, axis, open_width 속성 필요
    pose : (4,4) np.ndarray
        object -> world 변환
    sphere_radius : float
        center 표시 구의 반지름 (m)
    bar_length : float
        p1, p2에서 바깥으로 뻗는 막대 길이 (m)
    bar_radius : float
        막대 반지름 (m)
    gripper : optional
        .mesh, .T_grasp_gripper 속성 필요
    title : str
        viewer 창 제목
    """
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])

    # ---- 메쉬 ----
    mesh = graspable.mesh.copy()
    scene.add(pyrender.Mesh.from_trimesh(mesh), pose=pose)

    # ---- 월드 좌표축 ----
    axis_world = trimesh.creation.axis(
        transform=np.eye(4), axis_length=0.08, origin_size=0.003
    )
    scene.add(pyrender.Mesh.from_trimesh(axis_world, smooth=False), pose=np.eye(4))

    # ---- 오브젝트 pose 좌표축 ----
    axis_pose = trimesh.creation.axis(
        transform=pose, axis_length=0.03, origin_size=0.003
    )
    scene.add(pyrender.Mesh.from_trimesh(axis_pose, smooth=False), pose=np.eye(4))

    # ---- 색상 팔레트 ----
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

    # ---- 그래스프 ----
    for i, g in enumerate(grasps):
        v = g.axis / np.linalg.norm(g.axis)
        half_w = g.open_width / 2.0
        p1 = g.center + half_w * v
        p2 = g.center - half_w * v
        T = g.T_grasp_obj
        color = colors[i % len(colors)]

        # 그래스프 좌표축
        axis = trimesh.creation.axis(
            transform=pose @ T, axis_length=0.01, origin_size=0.001
        )
        scene.add(pyrender.Mesh.from_trimesh(axis, smooth=False), pose=np.eye(4))

        # ---- p1, p2 막대 (axis와 평행, 바깥쪽 방향) ----
        p1_outer = p1 + bar_length * v   # p1에서 +v 방향(바깥)
        p2_outer = p2 - bar_length * v   # p2에서 -v 방향(바깥)

        bar1 = trimesh.creation.cylinder(
            radius=bar_radius, segment=np.array([p1, p1_outer])
        )
        bar2 = trimesh.creation.cylinder(
            radius=bar_radius, segment=np.array([p2, p2_outer])
        )
        bar1.visual.face_colors = np.tile(color, (len(bar1.faces), 1))
        bar2.visual.face_colors = np.tile(color, (len(bar2.faces), 1))

        scene.add(pyrender.Mesh.from_trimesh(bar1, smooth=False), pose=pose)
        scene.add(pyrender.Mesh.from_trimesh(bar2, smooth=False), pose=pose)

    # ---- 조명 ----
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0)
    scene.add(light)

    pyrender.Viewer(scene, use_raymond_lighting=True, window_title=title)