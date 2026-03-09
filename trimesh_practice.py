import trimesh
import numpy as np

mesh = trimesh.load("/home/minsoo/bin_picking1/Minsoo_net/data/gripper/gripper.stl")
mesh.apply_scale(0.001)
# 원점에 축 표시 (RGB = XYZ)
axis = trimesh.creation.axis(origin_size=5, axis_length=50)
axis.apply_scale(0.001)
trans=np.array([[ 1.0,  0.0,  0.0,  0.0   ],
 [ 0.0,  0.0,  1.0,  0.056 ],
 [ 0.0, -1.0,  0.0,  0.0   ],
 [ 0.0,  0.0,  0.0,  1.0   ]])
new=axis.copy()
new=new.apply_transform(trans)
print('모델크기: ',mesh.extents)
# 함께 시각화
scene = trimesh.Scene([mesh, axis,new])
scene.show()

