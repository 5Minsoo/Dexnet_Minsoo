import trimesh
import numpy as np

# 1. 파일 로드 및 스케일 조정 (사용자 코드 유지)
mesh = trimesh.load("/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/object/bin.stl")
mesh2=trimesh.load("/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/object/PVCTT13.stl")
mesh.apply_scale(0.001)
# 2. Scene 생성 및 물체 추가
scene = trimesh.Scene()
scene.add_geometry([mesh2])

# 3. 시각화 창 띄우기
scene.show()

_,probs=mesh2.compute_stable_poses()
print(probs)