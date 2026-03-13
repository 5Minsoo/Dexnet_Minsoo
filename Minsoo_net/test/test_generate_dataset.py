from test_copy import GraspPipeline
import zarr
object_path='/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/object.stl'
pipeline=GraspPipeline(object_path,num_grasps=10,quality_threshold=0.02)


stable_poses,initial_grasps,collision_free,quality_grasps,qualities=pipeline.execute()

labels = []
depth=[]
center=[]
axis=[]
for grasp in collision_free:
    depth.append(grasp.center[2])
    center.append(grasp.center)
    axis.append(grasp.axis)
    if grasp in quality_grasps:
        idx = quality_grasps.index(grasp)
        labels.append(qualities[idx])
    else:
        labels.append(0.0)  

print(f"Center: {(center)}, Axis:{axis},Labels: {(labels)}, Depth: {(depth)}")

f=zarr.open('zarr_test',mode='w')
object=f.create_group('object')
object1=object.create_group('object1')
object1.create_array('stable_pose',data=stable_poses)

for i,pose in enumerate(stable_poses):
    stable_pose=object1.create_group(f'stable_pose{i}')
    stable_pose.attrs(pose)
    for j in range(collision_free):
        stable_pose.create_array(f'grasp{j}',data=(center,axis))

