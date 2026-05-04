import cadquery as cq

step_path = '/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/Intel Realsense D455.STEP'
obj_path  = '/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/D455.obj'

# STEP 읽기
shape = cq.importers.importStep(step_path)

# STL로 export (OBJ는 cadquery에서 직접 지원 X, STL이 호환성 좋음)
stl_path = '/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/D455.stl'
cq.exporters.export(shape, stl_path, tolerance=0.001)
print(f"✓ STL 저장: {stl_path}")

# STL → OBJ 변환 (trimesh 사용)
import trimesh
mesh = trimesh.load(stl_path)
mesh.export(obj_path)
print(f"✓ OBJ 저장: {obj_path}")