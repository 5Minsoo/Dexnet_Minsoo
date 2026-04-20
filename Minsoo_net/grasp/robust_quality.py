from Minsoo_net.grasp import PointGraspMetrics3D
import time
import numpy as np
class QuasiStaticGraspQualityRV():
    """ RV class for grasp quality on an object.

    Attributes
    ----------
    grasp_rv : :obj:`ParallelJawGraspPoseGaussianRV`
        random variable for gripper pose
    obj_rv : :obj:`GraspableObjectPoseGaussianRV`
        random variable for object pose
    params_rv : :obj:`ParamsGaussianRV`
        random variable for a set of grasp quality parameters
    quality_config : :obj:`GraspQualityConfig`
        parameters for grasp quality computation
    """
    def __init__(self, grasp_rv, obj_rv, params_rv):
        self.grasp_rv_ = grasp_rv
        self.obj_rv_ = obj_rv
        self.params_rv_ = params_rv # samples extra params for quality
        self.sample_count_ = 0

    def sample(self,size=1):
        """ Samples deterministic quasi-static point grasp quality metrics.

        Parameters
        ----------
        size : int
            number of samples to take
        """
        # sample grasp
        cur_time = time.time()
        grasp_sample = self.grasp_rv_.sample(size)
        grasp_time = time.time()

        # sample object
        obj_sample = self.obj_rv_.sample(size)
        obj_time = time.time()

        # sample params
        params_sample = self.params_rv_.sample(size)
        params_time = time.time()

        # compute deterministic quality
        start = time.time()
        quality=[]
        for i in range(size):
            try:
                q = PointGraspMetrics3D.grasp_quality(grasp_sample[i], obj_sample[i],params_sample[i])
                quality.append(q)
            except (np.linalg.LinAlgError, ValueError):
                continue

        quality_time = time.time()
        # print('Quality comp took %.3f sec' %(quality_time - start))
        self.sample_count_ = self.sample_count_ + 1
        return quality

    def expected_quality(self, size=1):
            """ 
            여러 번 샘플링하여 파지 품질의 기대값(평균)을 구하는 헬퍼 함수
            (논문에서 Robust Grasp Quality를 구할 때 주로 사용)
            """
            qualities = self.sample(size)
            if len(qualities) == 0:
                return 0.0
            return np.mean(qualities)