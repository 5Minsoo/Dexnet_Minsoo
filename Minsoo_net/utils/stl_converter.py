# scale_meshes.py
import trimesh
from pathlib import Path

folder_in  = Path("/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/object/FrankaPanda/FrankaPanda/meshes")
folder_out = Path("/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/object/Frankapanda")
scale = 0.001
exts = {".stl", ".obj", ".ply", ".dae"}

folder_out.mkdir(parents=True, exist_ok=True)

for f in folder_in.iterdir():
    if f.suffix.lower() not in exts:
        continue
    print(f"[Processing] {f.name}")
    mesh = trimesh.load(f, force="mesh")
    mesh.apply_scale(scale)
    mesh.export(folder_out / f.name)

print("Done.")