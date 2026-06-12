"""
Producer: Takes mesh folder. Generates GraspPipeline.execute. result as pkl file.
Consumer: render_tasks.py.

Assign--folder_num for parallel execution.
"""
import argparse
import pickle
import shutil
import time
from pathlib import Path

import trimesh
import yaml
import zarr

from Minsoo_net.grasp import GraspPipeline


PROJECT_ROOT = Path(__file__).parent.parent.resolve()           # .../Minsoo_net
TASKS_ROOT = PROJECT_ROOT / "data" / "tasks"
PENDING_DIR   = TASKS_ROOT / "pending"
STAGING_DIR   = TASKS_ROOT / "staging"
GENERATED_DIR = TASKS_ROOT / "generated"


def strip_grasp(g):
    g.contact_points = None
    return g


def parse_args():
    parser = argparse.ArgumentParser(description="Grasp task 생성 (Producer)")
    parser.add_argument("--folder_num", "-f", default="1", help="Frankapanda 하위 폴더 번호")
    parser.add_argument("--mode", "-m", default="continue",
                        choices=["continue", "restart"],
                        help="restart: 해당 폴더 mesh의 generated 마커/staging 정리 후 재생성")
    return parser.parse_args()


def load_config():
    config_path = PROJECT_ROOT / "config" / "master_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def list_mesh_files(folder_num: str):
    mesh_dir = PROJECT_ROOT / "data" / "object" / "selected" / folder_num
    files = list(mesh_dir.rglob("*.obj")) + list(mesh_dir.rglob("*.stl"))
    return mesh_dir, files


def sync_zarr_done_markers(zarr_path: str, mesh_files):
    """ if attr['done'] == true, mark generated ok """
    store = zarr.open(zarr_path, mode="r")
    newly_marked = []
    for m in mesh_files:
        obj_name = m.stem
        marker = GENERATED_DIR / f"{obj_name}.ok"
        if marker.exists():
            continue
        if obj_name not in store:
            continue
        if store[obj_name].attrs.get("done", False):
            marker.touch()
            newly_marked.append(obj_name)
    return newly_marked


def publish_object(obj_name: str, obj_staging: Path) -> int:
    """staging/<obj>/pose*.pkl move to pending, mark ok. Returns num of moved"""
    moved = 0
    for f in sorted(obj_staging.iterdir()):
        if f.suffix != ".pkl":
            continue
        f.rename(PENDING_DIR / f"{obj_name}__{f.name}")
        moved += 1
    shutil.rmtree(obj_staging)

    # Safe regist using rename (os level)
    marker_tmp = PENDING_DIR / f"{obj_name}.READY.tmp"
    marker_tmp.touch()
    marker_tmp.rename(PENDING_DIR / f"{obj_name}.READY")
    return moved


def generate_for_object(mesh_path: Path, config: dict) -> bool:
    """Generate grasp result pkl, publish to pending. Returns if success"""
    obj_name = mesh_path.stem

    m = trimesh.load(str(mesh_path), force="mesh")
    if min(m.bounding_box_oriented.extents) < 0.005:
        print(f"[!] {obj_name} 너무 얇아서 제외")
        return False

    print(f"{obj_name} 로드중")
    try:
        pipeline = GraspPipeline(
            str(mesh_path),
            quality_threshold=config.get("quality_threshold", 0.002),
            num_grasps=config.get("num_grasps", 300),
            max_approach_angle_deg=config.get("max_angle_deg", 15),
            num_poses=config.get("num_stable_poses", 10),
        )
    except Exception as e:
        print(f"[!] {obj_name} pipeline 초기화 실패: {e}")
        return False

    obj_staging = STAGING_DIR / obj_name
    if obj_staging.exists():
        shutil.rmtree(obj_staging)
    obj_staging.mkdir(parents=True)

    t0 = time.time()
    pose_idx = 0
    try:
        for pose, failed_grasps, quality_grasps, quality_scores in pipeline.execute(start_index=0):
            for g in list(quality_grasps) + list(failed_grasps):
                strip_grasp(g)

            payload = {
                "object": obj_name,
                "mesh_path": str(mesh_path),
                "pose_idx": pose_idx,
                "pose": pose,
                "quality_grasps": quality_grasps,
                "quality_scores": list(quality_scores),
                "failed_grasps": failed_grasps,
            }
            # Safe dump using rename
            out = obj_staging / f"pose{pose_idx}.pkl"
            tmp = out.with_suffix(".pkl.tmp")
            with open(tmp, "wb") as f:
                pickle.dump(payload, f)
            tmp.rename(out)
            print(f"  saved {obj_name} pose{pose_idx}: q={len(quality_grasps)} f={len(failed_grasps)}")
            pose_idx += 1
    except Exception as e:
        print(f"[!] {obj_name} 생성 중 예외, staging 보존하고 다음 물체로: {e}")
        import traceback; traceback.print_exc()
        return False

    moved = publish_object(obj_name, obj_staging)
    (GENERATED_DIR / f"{obj_name}.ok").touch()
    print(f"✓ {obj_name} task {moved}개 publish, 소요 {int(time.time()-t0)}초")
    return True


def main():
    args = parse_args()
    config = load_config()

    for d in (PENDING_DIR, STAGING_DIR, GENERATED_DIR):
        d.mkdir(parents=True, exist_ok=True)

    mesh_dir, mesh_files = list_mesh_files(args.folder_num)
    print(f"현재 path: {mesh_dir}")
    print(f"해당 물체 진행: {len(mesh_files)}")

    if args.mode == "restart":
        print("restart: 해당 폴더 mesh의 generated 마커 / staging 정리")
        for m in mesh_files:
            (GENERATED_DIR / f"{m.stem}.ok").unlink(missing_ok=True)
            so = STAGING_DIR / m.stem
            if so.exists():
                shutil.rmtree(so)
    else:
        zarr_path = config.get("zarr_path", "grasp_dataset.zarr")
        newly = sync_zarr_done_markers(zarr_path, mesh_files)
        if newly:
            print(f"zarr 에 이미 완료된 물체 마커 동기화 ({len(newly)}개): {newly}")

    skipped, todo = [], []
    for m in mesh_files:
        if (GENERATED_DIR / f"{m.stem}.ok").exists():
            skipped.append(m.stem)
        else:
            todo.append(m)
    if skipped:
        print(f"이미 task 생성 완료된 물체 건너뜀 ({len(skipped)}개): {skipped}")
    print(f"실제 진행할 물체: {len(todo)}개")

    for mesh_path in todo:
        generate_for_object(mesh_path, config)

    print("Producer 종료")


if __name__ == "__main__":
    main()
