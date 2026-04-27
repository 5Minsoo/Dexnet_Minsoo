import os
import trimesh
import numpy as np

MESH_EXTS = ('.obj', '.stl', '.ply', '.off', '.glb', '.gltf', '.dae', '.wrl')
MAX_MESHES = 100  # 처리할 최대 메쉬 개수

def print_obb_extents(folder, recursive=True, max_meshes=MAX_MESHES):
    """폴더 내 메쉬 파일들의 OBB extent 출력 (최대 max_meshes개)"""
    folder = os.path.abspath(folder)
    walker = os.walk(folder) if recursive else [(folder, [], os.listdir(folder))]
    
    results = []
    done = False
    for root, _, files in walker:
        if done:
            break
        for f in files:
            if not f.lower().endswith(MESH_EXTS):
                continue
            
            path = os.path.join(root, f)
            try:
                mesh = trimesh.load(path, force='mesh')
                
                # Scene이면 합치기
                if isinstance(mesh, trimesh.Scene):
                    mesh = trimesh.util.concatenate(
                        [g for g in mesh.geometry.values()]
                    )
                
                # OBB extent
                obb_extent = mesh.bounding_box_oriented.extents
                aabb_extent = mesh.extents
                
                rel_path = os.path.relpath(path, folder)
                results.append((rel_path, obb_extent, aabb_extent, len(mesh.faces)))
                
                # 100개 도달하면 중단
                if len(results) >= max_meshes:
                    done = True
                    break
                    
            except Exception as e:
                print(f"[FAIL] {f}: {e}")
    
    # 출력
    print(f"\n{'File':<40} {'OBB extent (m)':<35} {'AABB extent (m)':<35} {'Faces':>8}")
    print("-" * 120)
    for path, obb, aabb, nf in results:
        obb_str = f"[{obb[0]:.4f}, {obb[1]:.4f}, {obb[2]:.4f}]"
        aabb_str = f"[{aabb[0]:.4f}, {aabb[1]:.4f}, {aabb[2]:.4f}]"
        print(f"{path:<40} {obb_str:<35} {aabb_str:<35} {nf:>8}")
    
    # 통계
    if results:
        all_obb = np.array([r[1] for r in results])
        print(f"\n[OBB Statistics] (processed {len(results)} / max {max_meshes})")
        print(f"Min dim  - min: {all_obb.min(axis=1).min():.4f}, "
              f"max: {all_obb.min(axis=1).max():.4f}, "
              f"mean: {all_obb.min(axis=1).mean():.4f}")
        print(f"Max dim  - min: {all_obb.max(axis=1).min():.4f}, "
              f"max: {all_obb.max(axis=1).max():.4f}, "
              f"mean: {all_obb.max(axis=1).mean():.4f}")


if __name__ == '__main__':
    print_obb_extents(
        '/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/object/Frankapanda',
        recursive=True,
        max_meshes=100
    )