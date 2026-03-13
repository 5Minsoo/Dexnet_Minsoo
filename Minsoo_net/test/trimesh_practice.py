import trimesh
import numpy as np

mesh = trimesh.load("/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/gripper/gripper.stl")
mesh.apply_scale(0.001)
# 원점에 축 표시 (RGB = XYZ)
axis = trimesh.creation.axis(origin_size=5, axis_length=50)
axis.apply_scale(0.001)
trans=np.array([[ 1.0,  0.0,  0.0,  0.0   ],
 [ 0.0,  0.0,  1.0,  0.056 ],
 [ 0.0, -1.0,  0.0,  0.0   ],
 [ 0.0,  0.0,  0.0,  1.0   ]])

trans2=np.eye(4)
trans2[:3,3]=np.array([0.0,0.0,-0.056])
new=axis.copy()
new=new.apply_transform(trans)

# mesh_old=mesh.copy()
mesh.apply_transform(np.linalg.inv(trans))

    
print('모델크기: ',mesh.extents)
# 함께 시각화
scene = trimesh.Scene([mesh,axis,new])
scene.show()

