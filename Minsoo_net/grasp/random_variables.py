import numpy as np
import scipy
from scipy.spatial.transform import Rotation
import copy
from Minsoo_net.grasp import ParallelJawGrasp
import yaml
import sapien
class RandomVariables():
    def __init__(self):
        self.sigma_com_=1e-6
        self.sigma_scale_=1e-6
        self.sigma_rot_ = 1e-6
        self.sigma_trans_ = 1e-6
        self.sigma_approach_ = 1e-6
        self.sigma_friction_=1e-6
        self.mean_friction_=0.8


class GraspableObjectPoseGaussianRV(RandomVariables):
    def __init__(self, obj,config_yaml):
        super().__init__()
        self.obj_ = obj
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
        self.friction_rv_=scipy.stats.halfnorm(self.mean_friction_,self.sigma_friction_)
        
        # --- Metallic (Truncated Normal) ---
        a_met = (self.min_matalic_ - self.mean_matalic_) / self.sigma_matalic_
        b_met = (self.max_matalic_ - self.mean_matalic_) / self.sigma_matalic_
        self.metalic_rv_ = scipy.stats.truncnorm(a_met, b_met, loc=self.mean_matalic_, scale=self.sigma_matalic_)

        # --- Roughness (Truncated Normal) ---
        a_rough = (self.min_roughness - self.mean_roughness) / self.sigma_roughness
        b_rough = (self.max_roughness - self.mean_roughness) / self.sigma_roughness
        self.roughness_rv_ = scipy.stats.truncnorm(a_rough, b_rough, loc=self.mean_roughness, scale=self.sigma_roughness)

    def _parse_config(self, config_yaml):
        if config_yaml is not None:
            with open(config_yaml) as f:
                config = yaml.safe_load(f)
                self.mean_friction_=config.get("mean_friction",0.8)
                self.sigma_friction_ = config.get("sigma_friction", 0.1)

                self.sigma_matalic_ = config.get('sigma_metalic', 0.1)
                self.mean_matalic_ = config.get('mean_metalic', 0.9)
                self.min_matalic_ = config.get('min_metalic', 0.8)
                self.max_matalic_ = config.get('max_metalic', 1.0)
                
                self.mean_roughness = config.get('mean_roughness', 0.6)
                self.min_roughness = config.get('min_roughness', 0.6)
                self.sigma_roughness = config.get('sigma_roughness', 0.1)
                self.max_roughness = config.get('max_roughness', 1.0)    
                                               

    def sample(self,size=1):
        return self.friction_rv_.rvs(size)

    def sample_material(self,size=1):
        metallics = self.metalic_rv_.rvs(size)
        roughnesses = self.roughness_rv_.rvs(size)
        return metallics,roughnesses
    

if __name__ == "__main__":
    rv=ParamsGaussianRV('/home/minsoo/Dexnet_Minsoo/Minsoo_net/config/random_variables.yaml')
    print(rv.sample_material(100))