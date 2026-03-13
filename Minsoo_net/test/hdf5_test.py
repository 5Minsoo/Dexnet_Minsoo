import zarr
import numpy as np
my_list = np.array([1,2,3,4])
f=zarr.open('zarr_test',mode="w")
a=f.create_group('object1')

for i, data in enumerate(my_list):
    b=a.create_group(f"stable_pose{i}")
    for j in range(len(my_list)):
        b.create_array(f"grasp{j}",data=my_list[j])

