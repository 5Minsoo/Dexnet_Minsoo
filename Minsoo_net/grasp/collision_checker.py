"""
Collision checking using python-fcl
"""
import logging
import numpy as np
import fcl
import trimesh

from Minsoo_net.grasp.random_variables import GraspableObjectPoseGaussianRV, ParallelJawGraspPoseGaussianRV

def _load_mesh_as_bvh(mesh):
    """Load a mesh file from graspable class and return an fcl BVHModel."""
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int32)
    bvh = fcl.BVHModel()
    bvh.beginModel(len(faces), len(vertices))
    bvh.addSubModel(vertices, faces)
    bvh.endModel()
    return bvh


def _se3_to_fcl_transform(T):
    """Convert a 4x4 SE(3) numpy array to an fcl.Transform."""
    return fcl.Transform(T[:3, :3], T[:3, 3])


class CollisionChecker(object):
    """Wrapper for collision checking with python-fcl"""

    def __init__(self,tolerance=0.1,use_visual=False):
        self._geoms = {}
        self._objs = {}
        self._objs_tf = {}
        self._meshes = {}
        self.tolerance=1-tolerance
        self.use_visual=use_visual

    def remove_object(self, name):
        if name not in self._objs:
            return
        self._geoms.pop(name)
        self._objs.pop(name)
        self._objs_tf.pop(name)
        self._meshes.pop(name, None)

    def set_object(self, name, mesh, T_world_obj=None):
        """
        Parameters
        ----------
        name : str
        graspable : GraspableObject class
        T_world_obj : np.ndarray, shape (4,4)
            SE(3) transformation from object to world frame
        """
        mesh_object=mesh.copy()
        if name != "gripper":
            centroid = mesh_object.centroid
            mesh_object.vertices = centroid + (mesh_object.vertices - centroid) * self.tolerance
        bvh = _load_mesh_as_bvh(mesh_object)
        self._geoms[name] = bvh
        self._meshes[name] = mesh_object

        if T_world_obj is None:
            T_world_obj = np.eye(4)

        tf = _se3_to_fcl_transform(T_world_obj)
        self._objs[name] = fcl.CollisionObject(bvh, tf)
        self._objs_tf[name] = T_world_obj.copy()

    def set_transform(self, name, T_world_obj):
        """
        Parameters
        ----------
        name : str
        T_world_obj : np.ndarray, shape (4,4)
            SE(3) transformation from object to world frame
        """
        tf = _se3_to_fcl_transform(T_world_obj)
        self._objs[name].setTransform(tf)
        self._objs_tf[name] = T_world_obj.copy()

    def in_collision_single(self, target_name, names=None):
        if names is None:
            names = self._objs.keys()

        request = fcl.CollisionRequest()
        result = fcl.CollisionResult()
        target_obj = self._objs[target_name]

        for other_name in names:
            if other_name != target_name:
                ret = fcl.collide(self._objs[other_name], target_obj, request, result)
                if ret > 0:
                    logging.debug('Collision between: {0} and {1}'.format(other_name, target_name))
                    return True

        return False

    def in_collision(self, names=None, visualize=False):
        if names is None:
            names = list(self._objs.keys())

        request = fcl.CollisionRequest(enable_contact=True)
        result = fcl.CollisionResult()

        collision_pairs = []
        has_collision = False

        for i, name1 in enumerate(names):
            for name2 in names[i + 1:]:
                result = fcl.CollisionResult()
                ret = fcl.collide(self._objs[name1], self._objs[name2], request, result)
                if ret > 0:
                    # print('Collision between: {0} and {1}'.format(name1, name2))
                    collision_pairs.append((name1, name2))
                    has_collision = True

        if visualize:
            self._visualize_collision(names, collision_pairs, result)

        return has_collision

    def _visualize_collision(self, names, collision_pairs, result):
        scene = trimesh.Scene()
        colliding_names = set()
        for n1, n2 in collision_pairs:
            colliding_names.add(n1)
            colliding_names.add(n2)

        for name in names:
            # 바닥(table)은 Halfspace라 mesh가 없으므로 평면으로 시각화
            if name == 'table':
                T = self._objs_tf.get(name, np.eye(4))
                floor_mesh = trimesh.creation.box(extents=[0.5, 0.5, 0.001])
                floor_mesh.visual.face_colors = [200, 200, 200, 150]
                scene.add_geometry(floor_mesh, transform=T, node_name='table')
                # 테이블 좌표축
                table_axes = trimesh.creation.axis(origin_size=0.005, axis_length=0.08)
                scene.add_geometry(table_axes, transform=T, node_name='table_axes')
                continue

            if name not in self._meshes:
                continue
            mesh_copy = self._meshes[name].copy()
            T = self._objs_tf.get(name, np.eye(4))
            if name in colliding_names:
                mesh_copy.visual.face_colors = [255, 80, 80, 180]
            else:
                mesh_copy.visual.face_colors = [80, 200, 80, 180]
            scene.add_geometry(mesh_copy, transform=T, node_name=name)

            # 각 물체 좌표축
            obj_axes = trimesh.creation.axis(origin_size=0.003, axis_length=0.05)
            scene.add_geometry(obj_axes, transform=T, node_name=f'{name}_axes')

        # 월드 좌표축 (원점)
        world_axes = trimesh.creation.axis(origin_size=0.008, axis_length=0.12)
        scene.add_geometry(world_axes, transform=np.eye(4), node_name='world_axes')

        # 접촉점 표시
        contact_points = []
        for contact in result.contacts:
            contact_points.append(contact.pos)
        if contact_points:
            pts = np.array(contact_points)
            cloud = trimesh.PointCloud(pts, colors=[255, 255, 0, 255])
            scene.add_geometry(cloud, node_name='contact_points')

        scene.show()


class GraspCollisionChecker(CollisionChecker):
    """Collision checker that automatically handles grasp objects."""

    def __init__(self, gripper, use_visual=False):
        super().__init__(use_visual=use_visual)
        self._gripper = gripper
        self.set_object('gripper', self._gripper.mesh)

    @property
    def obj_names(self):
        return self._objs_tf.keys()

    def set_target_object(self, key):
        if key in self.obj_names:
            self._graspable_key = key

    def set_graspable_object(self, graspable, T_obj_world=None):
        if T_obj_world is None:
            T_obj_world = np.eye(4)
        self.set_object(graspable.key, graspable.model_name, T_obj_world)
        self.set_target_object(graspable.key)

    def add_graspable_object(self, graspable, T_obj_world=None):
        if T_obj_world is None:
            T_obj_world = np.eye(4)
        self.set_object(graspable.key, graspable.model_name, T_obj_world)

    def set_table(self):
        """
        Parameters
        ----------
        filename : str
        T_table_world : np.ndarray, shape (4,4)
            SE(3) pose of table w.r.t. world
        """
        T_table_world=np.eye(4)
        floor_geom=fcl.Halfspace(n=np.array([0.0,0.0,1.0]),d=0.0)
        T_table_world_fcl=_se3_to_fcl_transform(T_table_world)
        floor_col=fcl.CollisionObject(floor_geom,T_table_world_fcl)
        self._objs['table']=floor_col
        self._objs_tf['table']=T_table_world

    def grasp_in_collision(self, T_obj_gripper, key=None):
        """
        Parameters
        ----------
        T_obj_gripper : np.ndarray, shape (4,4)
            SE(3) pose of gripper w.r.t. object
        key : str

        Returns
        -------
        bool
        """
        T_world_gripper = self._objs_tf[key] @ T_obj_gripper
        T_mesh_gripper_inv = np.linalg.inv(self._gripper.T_grasp_gripper)
        T_world_mesh = T_world_gripper @ T_mesh_gripper_inv
        self.set_transform('gripper', T_world_mesh)

        return self.in_collision_single('gripper')

    def collides_along_approach(self, grasp, delta_approach=0.01,approach_dist=0.09,key=None):
        T_grasp_obj = grasp.T_grasp_obj
        grasp_approach_axis = T_grasp_obj[:3, 2]  # Z_axis
        logging.debug(f'T_grasp_obj: {T_grasp_obj}')

        collides = False
        cur_approach = 0.0
        while cur_approach <= approach_dist+1e-6 and not collides:
            T_approach_obj = T_grasp_obj.copy()
            T_approach_obj[:3, 3] -= cur_approach * grasp_approach_axis
            T_gripper_obj = T_approach_obj

            collides = self.grasp_in_collision(T_gripper_obj, key=key)
            cur_approach += delta_approach

        if collides and self.use_visual:
            self.in_collision(visualize=True)

        return collides

    def collision_prob(self, obj, grasp, key, num):                                                                                                     
        config_path = '/home/minsoo/Dexnet_Minsoo/Minsoo_net/config/master_config.yaml'
        grasp_rv = ParallelJawGraspPoseGaussianRV(grasp, config_path)                                                                                   
        obj_rv = GraspableObjectPoseGaussianRV(obj, config_path)
        grasp_samples = grasp_rv.sample(num)                                                                                                            
        T_samples = obj_rv.sample_transpose(num)
                                                                                                                                                        
        # 원래 pose 저장                                                                                                                                
        original_tfs = {name: self._objs_tf[name].copy() for name in self.obj_names}
                                                                                                                                                        
        collisions = 0                                                                                                                                
        for T_pert, g in zip(T_samples, grasp_samples):
            # 모든 물체에 perturbation 적용                                                                                                             
            for name in self.obj_names:
                if name != 'table' and name != 'gripper':
                    logging.debug(f'{name}에 Perturbation 적용')                                                                                               
                    self.set_transform(name, original_tfs[name] @ T_pert)
                                                                                                                                                        
            if self.collides_along_approach(g, key=key):
                collisions += 1

        # 원래 pose 복원
        for name, tf in original_tfs.items():
            if name != 'table' and name != 'gripper':                                                                                                   
                self.set_transform(name, tf)
                                                                                                                                                        
        return collisions / num
