import zarr
import numpy as np

f=zarr.open('zarr_test',"w")
f.create_group('object1')
f.create_group('/object1/antipodal_samples')
f.create_dataset('mesh',data='mesh_path')
print(f['mesh'][()])
