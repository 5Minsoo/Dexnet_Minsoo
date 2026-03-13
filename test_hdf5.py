import numpy as np
import zarr

f=zarr.open('/home/minsoo/Dexnet_Minsoo/zarr_test','r')
print(f['mesh'][()])