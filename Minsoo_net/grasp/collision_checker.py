"""
Collision checking using python-fcl
"""
import logging
import numpy as np
import fcl
import trimesh


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

    def __init__(self):
        self._geoms = {}
        self._objs = {}
        self._objs_tf = {}
        self._meshes = {}

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
        bvh = _load_mesh_as_bvh(mesh)
        self._geoms[name] = bvh
        self._meshes[name] = mesh

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
                    print('Collision between: {0} and {1}'.format(name1, name2))
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
            if name not in self._meshes:
                continue
            mesh_copy = self._meshes[name].copy()
            T = self._objs_tf.get(name, np.eye(4))
            if name in colliding_names:
                mesh_copy.visual.face_colors = [255, 80, 80, 180]
            else:
                mesh_copy.visual.face_colors = [80, 200, 80, 180]
            scene.add_geometry(mesh_copy, transform=T, node_name=name)

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

    def __init__(self, gripper):
        super().__init__()
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

    def collides_along_approach(self, grasp, approach_dist, delta_approach, key=None):
        """
        Parameters
        ----------
        grasp : ParallelJawPtGrasp3D
        approach_dist : float
        delta_approach : float
        key : str

        Returns
        -------
        bool
        """
        T_grasp_obj = grasp.T_grasp_obj
        grasp_approach_axis = T_grasp_obj[:3, 2]  # x_axis

        collides = False
        cur_approach = 0.0
        while cur_approach <= approach_dist and not collides:
            T_approach_obj = T_grasp_obj.copy()
            T_approach_obj[:3, 3] -= cur_approach * grasp_approach_axis
            T_gripper_obj = T_approach_obj @ self._gripper.T_grasp_gripper

            collides = self.grasp_in_collision(T_gripper_obj, key=self._graspable_key)
            cur_approach += delta_approach

        return collides