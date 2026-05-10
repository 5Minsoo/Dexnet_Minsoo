"""
Consumer code: Reads pkl, Renders Images, marks done.
Supports multy execution.
"""
import argparse
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np
import yaml
import zarr

from Minsoo_net.grasp.rendering import GraspRenderer


PROJECT_ROOT = Path(__file__).parent.parent.resolve()       # .../Minsoo_net
TASKS_ROOT = PROJECT_ROOT / "data" / "tasks"
PENDING_DIR    = TASKS_ROOT / "pending"
PROCESSING_DIR = TASKS_ROOT / "processing"
DONE_DIR       = TASKS_ROOT / "done"

POSE_RE = re.compile(r"pose(\d+)\.pkl$")


def parse_args():
    parser = argparse.ArgumentParser(description="Grasp 렌더링 (Consumer)")
    parser.add_argument("--idle-timeout", type=int, default=180,
                        help="pending 비었을 때 대기 후 종료(초)")
    parser.add_argument("--poll-interval", type=int, default=3,
                        help="pending 비었을 때 재확인 간격(초)")
    parser.add_argument("--recover", action="store_true",
                        help="시작 시 죽은 워커의 processing/* 를 pending 으로 되돌림")
    parser.add_argument("--max-objects", type=int, default=0,
                        help="N개 object 처리 후 자발적 종료 (0=무제한). "
                             "sapien GPU 메모리 누수 회피용 — bash 루프로 재시작.")
    return parser.parse_args()


def load_config():
    config_path = PROJECT_ROOT / "config" / "master_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_camera_positions(grasp, pose, offsets):
    T_world_gripper = pose @ grasp.T_grasp_obj
    z_axis = T_world_gripper[:3, 2]
    gripper_pos = T_world_gripper[:3, 3]
    cam_target = gripper_pos
    cam_positions = [gripper_pos - z_axis * d for d in offsets]
    return cam_positions, cam_target


def recover_dead_workers(my_id: str):
    """
    다른 워커가 남긴 processing/<id>/ 를 pending 으로 되돌린다.
    단, 해당 PID가 실제로 시스템에서 종료되었는지(/proc/pid 확인) 검사하여 살아있는 워커의 파일은 건드리지 않는다.
    """
    if not PROCESSING_DIR.exists():
        return
        
    for d in PROCESSING_DIR.iterdir():
        if not d.is_dir() or d.name == my_id:
            continue
            
        # 1. 폴더 이름(예: pid43239)에서 숫자(PID)만 추출
        try:
            target_pid = int(d.name.replace("pid", ""))
            
            # 2. 시스템에 해당 PID가 여전히 살아있는지 검사
            if Path(f"/proc/{target_pid}").exists():
                # 살아있으면 절대 건드리지 않고 넘어감
                continue
        except ValueError:
            pass # 폴더 이름이 pid숫자 형식이 아니면 무시
            
        # 3. 진짜 죽은 워커로 판명되면 파일 회수 시작
        for f in d.iterdir():
            f.rename(PENDING_DIR / f.name)
        try:
            d.rmdir()
        except OSError:
            pass
        print(f"[recover] 죽은 워커 {d.name} 의 task 를 pending 으로 구출 완료")


def claim_object(my_proc_dir: Path):
    """READY 마커가 있는 object 를 하나 골라 마커 + 모든 pose pkl 을 my_proc_dir 로 이동.
    반환: (obj_name, [pose pkl path...], marker_path) 또는 None."""
    markers = sorted(PENDING_DIR.glob("*.READY"))
    for marker in markers:
        obj_name = marker.name[:-len(".READY")]
        marker_dst = my_proc_dir / marker.name
        try:
            os.rename(marker, marker_dst)            # atomic claim
        except FileNotFoundError:
            continue                                  

        claimed = []
        for src in sorted(PENDING_DIR.glob(f"{obj_name}__pose*.pkl")):
            dst = my_proc_dir / src.name
            try:
                os.rename(src, dst)
                claimed.append(dst)
            except FileNotFoundError:
                continue

        def pose_idx(p):
            m = POSE_RE.search(p.name)
            return int(m.group(1)) if m else 0
        claimed.sort(key=pose_idx)
        return obj_name, claimed, marker_dst
    return None


def render_pose(renderer, task, img_ds, label_ds, z_ds,
                camera_offsets, crop_size, output_size, batch_flush):
    pose = task["pose"]
    quality_grasps = task["quality_grasps"]
    quality_scores = task["quality_scores"]
    failed_grasps  = task["failed_grasps"]

    renderer.set_stable_pose(pose)
    tmp_imgs, tmp_labels, tmp_z = [], [], []

    def flush():
        if not tmp_imgs:
            return
        img_ds.append(np.array(tmp_imgs))
        label_ds.append(np.array(tmp_labels))
        z_ds.append(np.array(tmp_z))
        tmp_imgs.clear(); tmp_labels.clear(); tmp_z.clear()

    groups = [(quality_grasps, quality_scores),
              (failed_grasps, [0.0] * len(failed_grasps))]

    for grasps, labels in groups:
        for grasp, label in zip(grasps, labels):
            cam_poses, cam_target = get_camera_positions(grasp, pose, offsets=camera_offsets)
            metalic, roughness = renderer.sample_material()
            renderer.set_material(metalic=metalic, roughness=roughness)
            for cam_pos in cam_poses:
                depth = renderer.render(camera_pos=cam_pos, target_pos=cam_target)
                origin = [0, 0, 0]
                center = (pose @ np.append(grasp.center, 1.0))[:3]
                axis   = (pose @ np.append(grasp.axis, 1.0))[:3]
                image_point = renderer.world_to_pixel([origin, center, axis])
                grasp_depth = (renderer.get_extrinsic() @ np.append(center, 1.0))[2]
                cropped = GraspRenderer.crop_grasp_image(
                    depth, image_point[1], image_point[2] - image_point[0],
                    crop_size=crop_size, output_size=output_size,
                )
                tmp_imgs.append(cropped)
                tmp_labels.append(label)
                tmp_z.append(grasp_depth)
                if len(tmp_imgs) >= batch_flush:
                    flush()
    flush()


def process_object(store, obj_name, task_files, marker_path, config,
                   output_size, crop_size, batch_size, batch_flush):
    """  Read obejcts pose pkl, render, write on zarr file  """
    # move task to done, if marked done in zarr
    if obj_name in store and store[obj_name].attrs.get("done", False):
        print(f"[skip] {obj_name} 은 zarr 에 이미 done. 렌더링 생략.")
        for tf in task_files:
            tf.rename(DONE_DIR / tf.name)
        marker_path.rename(DONE_DIR / marker_path.name)
        return True

    # Get mesh path from first file, init renderer
    with open(task_files[0], "rb") as f:
        first = pickle.load(f)
    mesh_path = first["mesh_path"]

    try:
        renderer = GraspRenderer(mesh_path)
    except Exception as e:
        print(f"[!] {obj_name} renderer 초기화 실패, task 반납: {e}")
        for tf in task_files:
            tf.rename(PENDING_DIR / tf.name)
        marker_path.rename(PENDING_DIR / marker_path.name)
        return False

    co = config["cam_offset"]
    camera_offsets = np.linspace(co["start"], co["stop"], co["num"])

    obj_group = store.require_group(obj_name)

    for tf in task_files:
        with open(tf, "rb") as f:
            task = pickle.load(f)
        pose_idx = task["pose_idx"]

        pose_group = obj_group.require_group(f"pose{pose_idx}")
        img_ds = pose_group.create_array(
            "images", shape=(0, output_size, output_size),
            chunks=(batch_size, output_size, output_size),
            dtype="float32", overwrite=True,
        )
        label_ds = pose_group.create_array(
            "labels", shape=(0,), chunks=(batch_size,),
            dtype="float32", overwrite=True,
        )
        z_ds = pose_group.create_array(
            "gripper_depth", shape=(0,), chunks=(batch_size,),
            dtype="float32", overwrite=True,
        )

        t0 = time.time()
        render_pose(renderer, task, img_ds, label_ds, z_ds,
                    camera_offsets, crop_size, output_size, batch_flush)
        print(f"  {obj_name} pose{pose_idx} 렌더링 {int(time.time()-t0)}초")
        tf.rename(DONE_DIR / tf.name)

    obj_group.attrs["done"] = True
    marker_path.rename(DONE_DIR / marker_path.name)
    del renderer
    return True


def main():
    args = parse_args()
    config = load_config()

    for d in (PENDING_DIR, PROCESSING_DIR, DONE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    my_id = f"pid{os.getpid()}"
    my_proc_dir = PROCESSING_DIR / my_id
    my_proc_dir.mkdir(exist_ok=True)
    print(f"[worker {my_id}] start")

    if args.recover:
        recover_dead_workers(my_id)

    output_size = 32
    crop_size = 96
    batch_size = 2048
    batch_flush = 512

    zarr_path = config.get("zarr_path", "grasp_dataset.zarr")
    store = zarr.open(zarr_path, mode="a")
    if "config" not in store.attrs:
        store.attrs["config"] = config

    last_seen = time.time()
    while True:
        claim = claim_object(my_proc_dir)
        if claim is None:
            if time.time() - last_seen > args.idle_timeout:
                print(f"[worker {my_id}] idle timeout, exit")
                break
            time.sleep(args.poll_interval)
            continue
        last_seen = time.time()

        obj_name, task_files, marker_path = claim
        print(f"[worker {my_id}] claimed {obj_name}: {len(task_files)} poses")
        ok = process_object(
            store, obj_name, task_files, marker_path, config,
            output_size, crop_size, batch_size, batch_flush,
        )
        if ok:
            print(f"[worker {my_id}] ✓ {obj_name} 완료")

    if my_proc_dir.exists() and not any(my_proc_dir.iterdir()):
        my_proc_dir.rmdir()
    print(f"[worker {my_id}] exit")


if __name__ == "__main__":
    main()
