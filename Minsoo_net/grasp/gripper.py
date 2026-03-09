import numpy as np
import trimesh
import yaml

class RobotGripper():
    def __init__(self, name, yaml_path):
        with open(yaml_path) as f:
            config=yaml.safe_load(f)
        self.name = name
        self.mesh_filepath = config[name]['mesh_path']
        self.mesh = trimesh.load(self.mesh_filepath) 
        self.mesh=self.mesh.apply_scale(0.001)
        self.T_mesh_gripper = np.array(config[name]['T_mesh_gripper'])
        self.T_grasp_gripper=self.T_mesh_gripper