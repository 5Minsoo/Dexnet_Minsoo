import numpy as np
import scipy
from scipy.spatial.transform import Rotation
import copy
from Minsoo_net.grasp import ParallelJawGrasp
import yaml
class RandomVariables():
    def __init__(self):
        self.sigma_com_=1e-6
        self.sigma_scale_=1e-6
        self.sigma_rot_ = 1e-6
        self.sigma_trans_ = 1e-6
        self.sigma_approach_ = 1e-6
        self.sigma_friction_=1e-6


class GraspableObjectPoseGaussianRV(RandomVariables):
    def __init__(self, obj, mean_T_obj_world,config_yaml):
        super().__init__()
        self.obj_ = obj
        self.mean_T_obj_world_ = mean_T_obj_world
        self.config=self._parse_config(config_yaml)
        self.s_rv_ = scipy.stats.norm(1.0, self.sigma_scale_)
        self.t_rv_ = scipy.stats.multivariate_normal(np.zeros(3), self.sigma_trans_**2)
        self.r_rv_ = scipy.stats.multivariate_normal(np.zeros(3), self.sigma_rot_**2)
        self.com_rv_ = scipy.stats.multivariate_normal(np.zeros(3), self.sigma_com_**2)
    
    def _parse_config(self,config_yaml):
        if config_yaml is not None:
            with open(config_yaml) as f:
                config=yaml.safe_load(f)
            self.sigma_trans_=config['sigma_translation']
            self.sigma_rot_=config['sigma_rotation']
            self.sigma_scale_=config['sigma_scale']
            self.sigma_com_=config['sigma_center_of_mass']


    def sample(self, size=1):
        """ Sample random variables from the model.

        Parameters
        ----------
        size : int
            number of sample to take
        
        Returns
        -------
        :obj:`list` of :obj:`GraspableObject3D`
            sampled graspable objects from the pose random variable
        """
        samples = []
        for i in range(size):
            num_consecutive_failures = 0
            prev_len = len(samples)
            while len(samples) == prev_len:
                try:
                    # sample random pose
                    r = self.r_rv_.rvs(size=1)
                    R=Rotation.from_rotvec(r).as_matrix()
                    c = self.com_rv_.rvs(size=1)
                    t = self.t_rv_.rvs(size=1)
                    T= np.eye(4)
                    T[:3,3]=t.T
                    T[:3,:3]=R
                    com=R@c
                    sample_obj=copy.deepcopy(self.obj_)
                    sample_obj.mesh.apply_transform(T)
                    sample_obj.center_mass=com+sample_obj.mesh.center_mass
                    # transform object by pose
                    samples.append(sample_obj)

                except Exception as e:
                    num_consecutive_failures += 1
                    if num_consecutive_failures > 3:
                        raise

        # not a list if only 1 sample
        if size == 1 and len(samples) > 0:
            return samples[0]
        return samples
        

class ParallelJawGraspPoseGaussianRV(RandomVariables):
    """ Random variable for sampling grasps in different poses, to model uncertainty in robot repeatability

    Attributes
    ----------
    t_rv : :obj:`scipy.stats.multivariate_normal`
        multivariate Gaussian random variable for grasp translation
    r_xi_rv : :obj:`scipy.stats.multivariate_normal`
        multivariate Gaussian random variable of grasp rotations over the Lie Algebra
    R_sample_sigma : 3x3 :obj:`numpy.ndarray`
        rotation from the sampling reference frame to the random variable reference frame (e.g. for use with uncertainty only in the plane of the table)
    """
    def __init__(self, grasp, config):
        super().__init__()
        self.grasp_ = grasp
        self._parse_config(config)

        self.t_rv_ = scipy.stats.multivariate_normal(np.zeros(3), self.sigma_trans_**2)
        self.r_rv_ = scipy.stats.multivariate_normal(np.zeros(3), self.sigma_rot_**2)
        self.approach_rv_ = scipy.stats.norm(0.0, self.sigma_approach_)

    def _parse_config(self,config_yaml):
        if config_yaml is not None:
            with open(config_yaml) as f:
                config=yaml.safe_load(f)
            self.sigma_trans_= config['sigma_translation']
            self.sigma_rot_=config['sigma_rotation']
            self.sigma_approach_=config['sigma_gripper_approach_angle']

    def sample(self, size=1):
        samples = []
        for i in range(size):
            # sample random pose
            r = self.r_rv_.rvs(size=1)
            R=Rotation.from_rotvec(r).as_matrix()
            t =self.t_rv_.rvs(size=1).T
            sample_grasp=copy.deepcopy(self.grasp_)
            sample_grasp.center+=t
            sample_grasp.axis=R@sample_grasp.axis
            samples.append(sample_grasp)

        if size == 1:
            return samples[0]
        return samples

class ParamsGaussianRV(RandomVariables):
    def __init__(self,config_yaml):
        super().__init__()
        self._parse_config(config_yaml)
        self.friction_rv_=scipy.stats.halfnorm(0.5,self.sigma_friction_)

    def _parse_config(self,config_yaml):
        if config_yaml is not None:
            with open(config_yaml) as f:
                config=yaml.safe_load(f)
                self.sigma_friction_=config["sigma_friction"]
    
    def sample(self,size=1):
        return self.friction_rv_.rvs(size)
