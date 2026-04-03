import trimesh
mesh=trimesh.load('/home/minsoo/Dexnet_Minsoo/Minsoo_net/data/object/RDOC40-35.stl')
scene=trimesh.Scene()
scene.add_geometry([mesh])
scene.show()