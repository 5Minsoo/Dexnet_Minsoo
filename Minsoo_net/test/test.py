import trimesh
import numpy as np
import zarr

f=zarr.open('/home/minsoo/Dexnet_Minsoo/grasp_dataset_260000.zarr')
object=f['bin']['pose0']['images'][0]
print(object.shape)