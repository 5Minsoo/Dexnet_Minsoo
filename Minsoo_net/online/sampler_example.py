import numpy as np
class AntipodalDepthImageGraspSampler():
    """Grasp sampler for antipodal point pairs from depth image gradients.

    Notes
    -----
    Required configuration parameters are specified in Other Parameters.

    Other Parameters
    ----------------
    gripper_width : float
        Width of the gripper, in meters.
    friction_coef : float
        Friction coefficient for 2D force closure.
    depth_grad_thresh : float
        Threshold for depth image gradients to determine edge points for
        sampling.
    depth_grad_gaussian_sigma : float
        Sigma used for pre-smoothing the depth image for better gradients.
    downsample_rate : float
        Factor to downsample the depth image by before sampling grasps.
    max_rejection_samples : int
        Ceiling on the number of grasps to check in antipodal grasp rejection
        sampling.
    max_dist_from_center : int
        Maximum allowable distance of a grasp from the image center.
    min_grasp_dist : float
        Threshold on the grasp distance.
    angle_dist_weight : float
        Amount to weight the angle difference in grasp distance computation.
    depth_samples_per_grasp : int
        Number of depth samples to take per grasp.
    min_depth_offset : float
        Offset from the minimum depth at the grasp center pixel to use in depth
        sampling.
    max_depth_offset : float
        Offset from the maximum depth across all edges.
    depth_sample_win_height : float
        Height of a window around the grasp center pixel used to determine min
        depth.
    depth_sample_win_height : float
        Width of a window around the grasp center pixel used to determine min
        depth.
    depth_sampling_mode : str
        Name of depth sampling mode (uniform, min, max).
    """

    def __init__(self, config, gripper_width=np.inf):
        # Init superclass.
        ImageGraspSampler.__init__(self, config)

        # Antipodality params.
        self._gripper_width = self._config["gripper_width"]
        self._friction_coef = self._config["friction_coef"]
        self._depth_grad_thresh = self._config["depth_grad_thresh"]
        self._depth_grad_gaussian_sigma = self._config[
            "depth_grad_gaussian_sigma"]
        self._downsample_rate = self._config["downsample_rate"]
        self._rescale_factor = 1.0 / self._downsample_rate
        self._max_rejection_samples = self._config["max_rejection_samples"]

        self._min_num_edge_pixels = 0
        if "min_num_edge_pixels" in self._config:
            self._min_num_edge_pixels = self._config["min_num_edge_pixels"]

        # Distance thresholds for rejection sampling.
        self._max_dist_from_center = self._config["max_dist_from_center"]
        self._min_dist_from_boundary = self._config["min_dist_from_boundary"]
        self._min_grasp_dist = self._config["min_grasp_dist"]
        self._angle_dist_weight = self._config["angle_dist_weight"]

        # Depth sampling params.
        self._depth_samples_per_grasp = max(
            self._config["depth_samples_per_grasp"], 1)
        self._min_depth_offset = self._config["min_depth_offset"]
        self._max_depth_offset = self._config["max_depth_offset"]
        self._h = self._config["depth_sample_win_height"]
        self._w = self._config["depth_sample_win_width"]
        self._depth_sampling_mode = self._config["depth_sampling_mode"]

        # Perturbation.
        self._grasp_center_sigma = 0.0
        if "grasp_center_sigma" in self._config:
            self._grasp_center_sigma = self._config["grasp_center_sigma"]
        self._grasp_angle_sigma = 0.0
        if "grasp_angle_sigma" in self._config:
            self._grasp_angle_sigma = np.deg2rad(
                self._config["grasp_angle_sigma"])

    def _surface_normals(self, depth_im, edge_pixels):
        """Return an array of the surface normals at the edge pixels."""
        # Compute the gradients.
        grad = np.gradient(depth_im.data.astype(np.float32))

        # Compute surface normals.
        normals = np.zeros([edge_pixels.shape[0], 2])
        for i, pix in enumerate(edge_pixels):
            dx = grad[1][pix[0], pix[1]]
            dy = grad[0][pix[0], pix[1]]
            normal_vec = np.array([dy, dx])
            if np.linalg.norm(normal_vec) == 0:
                normal_vec = np.array([1, 0])
            normal_vec = normal_vec / np.linalg.norm(normal_vec)
            normals[i, :] = normal_vec

        return normals

    def _sample_depth(self, min_depth, max_depth):
        """Samples a depth value between the min and max."""
        depth_sample = max_depth
        if self._depth_sampling_mode == DepthSamplingMode.UNIFORM:
            depth_sample = min_depth + (max_depth -
                                        min_depth) * np.random.rand()
        elif self._depth_sampling_mode == DepthSamplingMode.MIN:
            depth_sample = min_depth
        return depth_sample

    def _sample(self,
                image,
                camera_intr,
                num_samples,
                segmask=None,
                visualize=False,
                constraint_fn=None):
        """Sample a set of 2D grasp candidates from a depth image.

        Parameters
        ----------
        image : :obj:`autolab_core.RgbdImage` or :obj:`autolab_core.DepthImage` or :obj:`autolab_core.GdImage`  # noqa: E501
            RGB-D or Depth image to sample from.
        camera_intr : :obj:`autolab_core.CameraIntrinsics`
            Intrinsics of the camera that captured the images.
        num_samples : int
            Number of grasps to sample
        segmask : :obj:`autolab_core.BinaryImage`
            Binary image segmenting out the object of interest.
        visualize : bool
            Whether or not to show intermediate samples (for debugging).
        constraint_fn : :obj:`GraspConstraintFn`
            Constraint function to apply to grasps.

        Returns
        -------
        :obj:`list` of :obj:`Grasp2D`
            List of 2D grasp candidates.
        """
        if isinstance(image, RgbdImage) or isinstance(image, GdImage):
            depth_im = image.depth
        elif isinstance(image, DepthImage):
            depth_im = image
        else:
            raise ValueError(
                "image type must be one of [RgbdImage, DepthImage, GdImage]")

        # Sample antipodal pairs in image space.
        grasps = self._sample_antipodal_grasps(depth_im,
                                               camera_intr,
                                               num_samples,
                                               segmask=segmask,
                                               visualize=visualize,
                                               constraint_fn=constraint_fn)
        return grasps

    def _sample_antipodal_grasps(self,
                                 depth_im,
                                 camera_intr,
                                 num_samples,
                                 segmask=None,
                                 visualize=False,
                                 constraint_fn=None):
        """Sample a set of 2D grasp candidates from a depth image by finding
        depth edges, then uniformly sampling point pairs and keeping only
        antipodal grasps with width less than the maximum allowable.

        Parameters
        ----------
        depth_im : :obj:"autolab_core.DepthImage"
            Depth image to sample from.
        camera_intr : :obj:`autolab_core.CameraIntrinsics`
            Intrinsics of the camera that captured the images.
        num_samples : int
            Number of grasps to sample.
        segmask : :obj:`autolab_core.BinaryImage`
            Binary image segmenting out the object of interest.
        visualize : bool
            Whether or not to show intermediate samples (for debugging).
        constraint_fn : :obj:`GraspConstraintFn`
            Constraint function to apply to grasps.

        Returns
        -------
        :obj:`list` of :obj:`Grasp2D`
            List of 2D grasp candidates.
        """
        # Compute edge pixels.
        edge_start = time()
        depth_im = depth_im.apply(snf.gaussian_filter,
                                  sigma=self._depth_grad_gaussian_sigma)
        scale_factor = self._rescale_factor
        depth_im_downsampled = depth_im.resize(scale_factor)
        depth_im_threshed = depth_im_downsampled.threshold_gradients(
            self._depth_grad_thresh)
        edge_pixels = (1.0 / scale_factor) * depth_im_threshed.zero_pixels()
        edge_pixels = edge_pixels.astype(np.int16)

        depth_im_mask = depth_im.copy()
        if segmask is not None:
            edge_pixels = np.array(
                [p for p in edge_pixels if np.any(segmask[p[0], p[1]] > 0)])
            depth_im_mask = depth_im.mask_binary(segmask)

        # Re-threshold edges if there are too few.
        if edge_pixels.shape[0] < self._min_num_edge_pixels:
            self._logger.info("Too few edge pixels!")
            depth_im_threshed = depth_im.threshold_gradients(
                self._depth_grad_thresh)
            edge_pixels = depth_im_threshed.zero_pixels()
            edge_pixels = edge_pixels.astype(np.int16)
            depth_im_mask = depth_im.copy()
            if segmask is not None:
                edge_pixels = np.array([
                    p for p in edge_pixels if np.any(segmask[p[0], p[1]] > 0)
                ])
                depth_im_mask = depth_im.mask_binary(segmask)

        num_pixels = edge_pixels.shape[0]
        self._logger.debug("Depth edge detection took %.3f sec" %
                           (time() - edge_start))
        self._logger.debug("Found %d edge pixels" % (num_pixels))

        # Compute point cloud.
        point_cloud_im = camera_intr.deproject_to_image(depth_im_mask)

        # Compute_max_depth.
        depth_data = depth_im_mask.data[depth_im_mask.data > 0]
        if depth_data.shape[0] == 0:
            return []

        min_depth = np.min(depth_data) + self._min_depth_offset
        max_depth = np.max(depth_data) + self._max_depth_offset

        # Compute surface normals.
        normal_start = time()
        edge_normals = self._surface_normals(depth_im, edge_pixels)
        self._logger.debug("Normal computation took %.3f sec" %
                           (time() - normal_start))

        if visualize:
            edge_pixels = edge_pixels[::2, :]
            edge_normals = edge_normals[::2, :]

            vis.figure()
            vis.subplot(1, 3, 1)
            vis.imshow(depth_im)
            if num_pixels > 0:
                vis.scatter(edge_pixels[:, 1], edge_pixels[:, 0], s=2, c="b")

            X = [pix[1] for pix in edge_pixels]
            Y = [pix[0] for pix in edge_pixels]
            U = [3 * pix[1] for pix in edge_normals]
            V = [-3 * pix[0] for pix in edge_normals]
            plt.quiver(X,
                       Y,
                       U,
                       V,
                       units="x",
                       scale=0.25,
                       width=0.5,
                       zorder=2,
                       color="r")
            vis.title("Edge pixels and normals")

            vis.subplot(1, 3, 2)
            vis.imshow(depth_im_threshed)
            vis.title("Edge map")

            vis.subplot(1, 3, 3)
            vis.imshow(segmask)
            vis.title("Segmask")
            vis.show()

        # Exit if no edge pixels.
        if num_pixels == 0:
            return []

        # Form set of valid candidate point pairs.
        pruning_start = time()
        max_grasp_width_px = Grasp2D(Point(np.zeros(2)),
                                     0.0,
                                     min_depth,
                                     width=self._gripper_width,
                                     camera_intr=camera_intr).width_px
        normal_ip = edge_normals.dot(edge_normals.T)
        dists = ssd.squareform(ssd.pdist(edge_pixels))
        valid_indices = np.where(
            (normal_ip < -np.cos(np.arctan(self._friction_coef)))
            & (dists < max_grasp_width_px) & (dists > 0.0))
        valid_indices = np.c_[valid_indices[0], valid_indices[1]]
        self._logger.debug("Normal pruning %.3f sec" %
                           (time() - pruning_start))

        # Raise exception if no antipodal pairs.
        num_pairs = valid_indices.shape[0]
        if num_pairs == 0:
            return []

        # Prune out grasps.
        contact_points1 = edge_pixels[valid_indices[:, 0], :]
        contact_points2 = edge_pixels[valid_indices[:, 1], :]
        contact_normals1 = edge_normals[valid_indices[:, 0], :]
        contact_normals2 = edge_normals[valid_indices[:, 1], :]
        v = contact_points1 - contact_points2
        v_norm = np.linalg.norm(v, axis=1)
        v = v / np.tile(v_norm[:, np.newaxis], [1, 2])
        ip1 = np.sum(contact_normals1 * v, axis=1)
        ip2 = np.sum(contact_normals2 * (-v), axis=1)
        ip1[ip1 > 1.0] = 1.0
        ip1[ip1 < -1.0] = -1.0
        ip2[ip2 > 1.0] = 1.0
        ip2[ip2 < -1.0] = -1.0
        beta1 = np.arccos(ip1)
        beta2 = np.arccos(ip2)
        alpha = np.arctan(self._friction_coef)
        antipodal_indices = np.where((beta1 < alpha) & (beta2 < alpha))[0]

        # Raise exception if no antipodal pairs.
        num_pairs = antipodal_indices.shape[0]
        if num_pairs == 0:
            return []
        sample_size = min(self._max_rejection_samples, num_pairs)
        grasp_indices = np.random.choice(antipodal_indices,
                                         size=sample_size,
                                         replace=False)
        self._logger.debug("Grasp comp took %.3f sec" %
                           (time() - pruning_start))

        # Compute grasps.
        sample_start = time()
        k = 0
        grasps = []
        while k < sample_size and len(grasps) < num_samples:
            grasp_ind = grasp_indices[k]
            p1 = contact_points1[grasp_ind, :]
            p2 = contact_points2[grasp_ind, :]
            n1 = contact_normals1[grasp_ind, :]
            n2 = contact_normals2[grasp_ind, :]
            #            width = np.linalg.norm(p1 - p2)
            k += 1

            # Compute center and axis.
            grasp_center = (p1 + p2) // 2
            grasp_axis = p2 - p1
            grasp_axis = grasp_axis / np.linalg.norm(grasp_axis)
            grasp_theta = np.pi / 2
            if grasp_axis[1] != 0:
                grasp_theta = np.arctan2(grasp_axis[0], grasp_axis[1])
            grasp_center_pt = Point(np.array(
                [grasp_center[1], grasp_center[0]]),
                                    frame=camera_intr.frame)

            # Compute grasp points in 3D.
            x1 = point_cloud_im[p1[0], p1[1]]
            x2 = point_cloud_im[p2[0], p2[1]]
            if np.linalg.norm(x2 - x1) > self._gripper_width:
                continue

            # Perturb.
            if self._grasp_center_sigma > 0.0:
                grasp_center_pt = grasp_center_pt + ss.multivariate_normal.rvs(
                    cov=self._grasp_center_sigma * np.diag(np.ones(2)))
            if self._grasp_angle_sigma > 0.0:
                grasp_theta = grasp_theta + ss.norm.rvs(
                    scale=self._grasp_angle_sigma)

            # Check center px dist from boundary.
            if (grasp_center[0] < self._min_dist_from_boundary
                    or grasp_center[1] < self._min_dist_from_boundary
                    or grasp_center[0] >
                    depth_im.height - self._min_dist_from_boundary
                    or grasp_center[1] >
                    depth_im.width - self._min_dist_from_boundary):
                continue

            # Sample depths.
            for i in range(self._depth_samples_per_grasp):
                # Get depth in the neighborhood of the center pixel.
                depth_win = depth_im.data[grasp_center[0] -
                                          self._h:grasp_center[0] + self._h,
                                          grasp_center[1] -
                                          self._w:grasp_center[1] + self._w]
                center_depth = np.min(depth_win)
                if center_depth == 0 or np.isnan(center_depth):
                    continue

                # Sample depth between the min and max.
                min_depth = center_depth + self._min_depth_offset
                max_depth = center_depth + self._max_depth_offset
                sample_depth = min_depth + (max_depth -
                                            min_depth) * np.random.rand()
                candidate_grasp = Grasp2D(grasp_center_pt,
                                          grasp_theta,
                                          sample_depth,
                                          width=self._gripper_width,
                                          camera_intr=camera_intr,
                                          contact_points=[p1, p2],
                                          contact_normals=[n1, n2])

                if visualize:
                    vis.figure()
                    vis.imshow(depth_im)
                    vis.grasp(candidate_grasp)
                    vis.scatter(p1[1], p1[0], c="b", s=25)
                    vis.scatter(p2[1], p2[0], c="b", s=25)
                    vis.show()

                grasps.append(candidate_grasp)

        # Return sampled grasps.
        self._logger.debug("Loop took %.3f sec" % (time() - sample_start))
        return grasps
    




class CrossEntropyRobustGraspingPolicy(GraspingPolicy):
    """Optimizes a set of grasp candidates in image space using the
    cross entropy method.

    Cross entropy method (CEM):
    (1) sample an initial set of candidates
    (2) sort the candidates
    (3) fit a GMM to the top P%
    (4) re-sample grasps from the distribution
    (5) repeat steps 2-4 for K iters
    (6) return the best candidate from the final sample set
    """

    def __init__(self, config, filters=None):
        """
        Parameters
        ----------
        config : dict
            Python dictionary of policy configuration parameters.
        filters : dict
            Python dictionary of functions to apply to filter invalid grasps.

        Notes
        -----
        Required configuration dictionary parameters are specified in Other
        Parameters.

        Other Parameters
        ----------------
        num_seed_samples : int
            Number of candidate to sample in the initial set.
        num_gmm_samples : int
            Number of candidates to sample on each resampling from the GMMs.
        num_iters : int
            Number of sample-and-refit iterations of CEM.
        gmm_refit_p : float
            Top p-% of grasps used for refitting.
        gmm_component_frac : float
            Percentage of the elite set size used to determine number of GMM
            components.
        gmm_reg_covar : float
            Regularization parameters for GMM covariance matrix, enforces
            diversity of fitted distributions.
        deterministic : bool, optional
            Whether to set the random seed to enforce deterministic behavior.
        gripper_width : float, optional
            Width of the gripper in meters.
        """
        GraspingPolicy.__init__(self, config)
        self._parse_config()
        self._filters = filters

        self._case_counter = 0

    def _parse_config(self):
        """Parses the parameters of the policy."""
        # Cross entropy method parameters.
        self._num_seed_samples = self.config["num_seed_samples"]
        self._num_gmm_samples = self.config["num_gmm_samples"]
        self._num_iters = self.config["num_iters"]
        self._gmm_refit_p = self.config["gmm_refit_p"]
        self._gmm_component_frac = self.config["gmm_component_frac"]
        self._gmm_reg_covar = self.config["gmm_reg_covar"]

        self._depth_gaussian_sigma = 0.0
        if "depth_gaussian_sigma" in self.config:
            self._depth_gaussian_sigma = self.config["depth_gaussian_sigma"]

        self._max_grasps_filter = 1
        if "max_grasps_filter" in self.config:
            self._max_grasps_filter = self.config["max_grasps_filter"]

        self._max_resamples_per_iteration = 100
        if "max_resamples_per_iteration" in self.config:
            self._max_resamples_per_iteration = self.config[
                "max_resamples_per_iteration"]

        self._max_approach_angle = np.inf
        if "max_approach_angle" in self.config:
            self._max_approach_angle = np.deg2rad(
                self.config["max_approach_angle"])

        # Gripper parameters.
        self._seed = None
        if self.config["deterministic"]:
            self._seed = GeneralConstants.SEED
        self._gripper_width = np.inf
        if "gripper_width" in self.config:
            self._gripper_width = self.config["gripper_width"]

        # Affordance map visualization.
        self._vis_grasp_affordance_map = False
        if "grasp_affordance_map" in self.config["vis"]:
            self._vis_grasp_affordance_map = self.config["vis"][
                "grasp_affordance_map"]

        self._state_counter = 0  # Used for logging state data.

    def select(self, grasps, q_values):
        """Selects the grasp with the highest probability of success.

        Can override for alternate policies (e.g. epsilon greedy).

        Parameters
        ----------
        grasps : list
            Python list of :obj:`gqcnn.grasping.Grasp2D` or
            :obj:`gqcnn.grasping.SuctionPoint2D` grasps to select from.
        q_values : list
            Python list of associated q-values.

        Returns
        -------
        :obj:`gqcnn.grasping.Grasp2D` or :obj:`gqcnn.grasping.SuctionPoint2D`
            Grasp with highest probability of success.
        """
        # Sort.
        self._logger.info("Sorting grasps")
        num_grasps = len(grasps)
        if num_grasps == 0:
            raise NoValidGraspsException("Zero grasps")
        grasps_and_predictions = zip(np.arange(num_grasps), q_values)
        grasps_and_predictions = sorted(grasps_and_predictions,
                                        key=lambda x: x[1],
                                        reverse=True)

        # Return top grasps.
        if self._filters is None:
            return grasps_and_predictions[0][0]

        # Filter grasps.
        self._logger.info("Filtering grasps")
        i = 0
        while i < self._max_grasps_filter and i < len(grasps_and_predictions):
            index = grasps_and_predictions[i][0]
            grasp = grasps[index]
            valid = True
            for filter_name, is_valid in self._filters.items():
                valid = is_valid(grasp)
                self._logger.debug("Grasp {} filter {} valid: {}".format(
                    i, filter_name, valid))
                if not valid:
                    valid = False
                    break
            if valid:
                return index
            i += 1
        raise NoValidGraspsException("No grasps satisfied filters")

    def _mask_predictions(self, pred_map, segmask):
        self._logger.info("Masking predictions...")
        seg_pred_mismatch_msg = ("Prediction map shape {} does not match shape"
                                 " of segmask {}.")
        assert pred_map.shape == segmask.shape, seg_pred_mismatch_msg.format(
            pred_map.shape, segmask.shape)
        preds_masked = np.zeros_like(pred_map)
        nonzero_ind = np.where(segmask > 0)
        preds_masked[nonzero_ind] = pred_map[nonzero_ind]
        return preds_masked

    def _gen_grasp_affordance_map(self, state, stride=1):
        self._logger.info("Generating grasp affordance map...")

        # Generate grasps at points to evaluate (this is just the interface to
        # `GraspQualityFunction`).
        crop_candidate_start_time = time()
        point_cloud_im = state.camera_intr.deproject_to_image(
            state.rgbd_im.depth)
        normal_cloud_im = point_cloud_im.normal_cloud_im()

        q_vals = []
        gqcnn_recep_h_half = self._grasp_quality_fn.gqcnn_recep_height // 2
        gqcnn_recep_w_half = self._grasp_quality_fn.gqcnn_recep_width // 2
        im_h = state.rgbd_im.height
        im_w = state.rgbd_im.width
        for i in range(gqcnn_recep_h_half - 1, im_h - gqcnn_recep_h_half,
                       stride):
            grasps = []
            for j in range(gqcnn_recep_w_half - 1, im_w - gqcnn_recep_w_half,
                           stride):
                # TODO(vsatish): Find a better way to find policy type.
                if self.config["sampling"]["type"] == "suction":
                    grasps.append(
                        SuctionPoint2D(Point(np.array([j, i])),
                                       axis=-normal_cloud_im[i, j],
                                       depth=state.rgbd_im.depth[i, j],
                                       camera_intr=state.camera_intr))
                else:
                    raise NotImplementedError(
                        "Parallel Jaw Grasp Affordance Maps Not Supported!")
            q_vals.extend(self._grasp_quality_fn(state, grasps))
        self._logger.info(
            "Generating crop grasp candidates took {} sec.".format(
                time() - crop_candidate_start_time))

        # Mask out predictions not in the segmask (we don't really care about
        # them).
        pred_map = np.array(q_vals).reshape(
            (im_h - gqcnn_recep_h_half * 2) // stride + 1,
            (im_w - gqcnn_recep_w_half * 2) // stride + 1)
        # TODO(vsatish): Don't access the raw data like this!
        tf_segmask = state.segmask.crop(im_h - gqcnn_recep_h_half * 2,
                                        im_w - gqcnn_recep_w_half * 2).resize(
                                            1.0 / stride,
                                            interp="nearest")._data.squeeze()
        if tf_segmask.shape != pred_map.shape:
            new_tf_segmask = np.zeros_like(pred_map)
            smaller_i = min(pred_map.shape[0], tf_segmask.shape[0])
            smaller_j = min(pred_map.shape[1], tf_segmask.shape[1])
            new_tf_segmask[:smaller_i, :smaller_j] = tf_segmask[:smaller_i, :
                                                                smaller_j]
            tf_segmask = new_tf_segmask
        pred_map_masked = self._mask_predictions(pred_map, tf_segmask)
        return pred_map_masked

    def _plot_grasp_affordance_map(self,
                                   state,
                                   affordance_map,
                                   stride=1,
                                   grasps=None,
                                   q_values=None,
                                   plot_max=True,
                                   title=None,
                                   scale=1.0,
                                   save_fname=None,
                                   save_path=None):
        gqcnn_recep_h_half = self._grasp_quality_fn.gqcnn_recep_height // 2
        gqcnn_recep_w_half = self._grasp_quality_fn.gqcnn_recep_width // 2
        im_h = state.rgbd_im.height
        im_w = state.rgbd_im.width

        # Plot.
        vis.figure()
        tf_depth_im = state.rgbd_im.depth.crop(
            im_h - gqcnn_recep_h_half * 2,
            im_w - gqcnn_recep_w_half * 2).resize(1.0 / stride,
                                                  interp="nearest")
        vis.imshow(tf_depth_im)
        plt.imshow(affordance_map, cmap=plt.cm.RdYlGn, alpha=0.3)
        if grasps is not None:
            grasps = copy.deepcopy(grasps)
            for grasp, q in zip(grasps, q_values):
                grasp.center.data[0] -= gqcnn_recep_w_half
                grasp.center.data[1] -= gqcnn_recep_h_half
                vis.grasp(grasp,
                          scale=scale,
                          show_center=False,
                          show_axis=True,
                          color=plt.cm.RdYlGn(q))
        if plot_max:
            affordance_argmax = np.unravel_index(np.argmax(affordance_map),
                                                 affordance_map.shape)
            plt.scatter(affordance_argmax[1],
                        affordance_argmax[0],
                        c="black",
                        marker=".",
                        s=scale * 25)
        if title is not None:
            vis.title(title)
        if save_path is not None:
            save_path = os.path.join(save_path, save_fname)
        vis.show(save_path)

    def action_set(self, state):
        """Plan a set of grasps with the highest probability of success on
        the given RGB-D image.

        Parameters
        ----------
        state : :obj:`RgbdImageState`
            Image to plan grasps on.

        Returns
        -------
        python list of :obj:`gqcnn.grasping.Grasp2D` or :obj:`gqcnn.grasping.SuctionPoint2D`  # noqa E501
            Grasps to execute.
        """
        # Check valid input.
        if not isinstance(state, RgbdImageState):
            raise ValueError("Must provide an RGB-D image state.")

        state_output_dir = None
        if self._logging_dir is not None:
            state_output_dir = os.path.join(
                self._logging_dir,
                "state_{}".format(str(self._state_counter).zfill(5)))
            if not os.path.exists(state_output_dir):
                os.makedirs(state_output_dir)
            self._state_counter += 1

        # Parse state.
        seed_set_start = time()
        rgbd_im = state.rgbd_im
        depth_im = rgbd_im.depth
        camera_intr = state.camera_intr
        segmask = state.segmask

        if self._depth_gaussian_sigma > 0:
            depth_im_filtered = depth_im.apply(
                snf.gaussian_filter, sigma=self._depth_gaussian_sigma)
        else:
            depth_im_filtered = depth_im
        point_cloud_im = camera_intr.deproject_to_image(depth_im_filtered)
        normal_cloud_im = point_cloud_im.normal_cloud_im()

        # Visualize grasp affordance map.
        if self._vis_grasp_affordance_map:
            grasp_affordance_map = self._gen_grasp_affordance_map(state)
            self._plot_grasp_affordance_map(state,
                                            grasp_affordance_map,
                                            title="Grasp Affordance Map",
                                            save_fname="affordance_map.png",
                                            save_path=state_output_dir)

        if "input_images" in self.config["vis"] and self.config["vis"][
                "input_images"]:
            vis.figure()
            vis.subplot(1, 2, 1)
            vis.imshow(depth_im)
            vis.title("Depth")
            vis.subplot(1, 2, 2)
            vis.imshow(segmask)
            vis.title("Segmask")
            filename = None
            if self._logging_dir is not None:
                filename = os.path.join(self._logging_dir, "input_images.png")
            vis.show(filename)

        # Sample grasps.
        self._logger.info("Sampling seed set")
        grasps = self._grasp_sampler.sample(
            rgbd_im,
            camera_intr,
            self._num_seed_samples,
            segmask=segmask,
            visualize=self.config["vis"]["grasp_sampling"],
            constraint_fn=self._grasp_constraint_fn,
            seed=self._seed)
        num_grasps = len(grasps)
        if num_grasps == 0:
            self._logger.warning("No valid grasps could be found")
            raise NoValidGraspsException()

        grasp_type = "parallel_jaw"
        if isinstance(grasps[0], SuctionPoint2D):
            grasp_type = "suction"
        elif isinstance(grasps[0], MultiSuctionPoint2D):
            grasp_type = "multi_suction"

        self._logger.info("Sampled %d grasps" % (len(grasps)))
        self._logger.info("Computing the seed set took %.3f sec" %
                          (time() - seed_set_start))

        # Iteratively refit and sample.
        for j in range(self._num_iters):
            self._logger.info("CEM iter %d" % (j))

            # Predict grasps.
            predict_start = time()
            q_values = self._grasp_quality_fn(state,
                                              grasps,
                                              params=self._config)
            self._logger.info("Prediction took %.3f sec" %
                              (time() - predict_start))

            # Sort grasps.
            resample_start = time()
            q_values_and_indices = zip(q_values, np.arange(num_grasps))
            q_values_and_indices = sorted(q_values_and_indices,
                                          key=lambda x: x[0],
                                          reverse=True)

            if self.config["vis"]["grasp_candidates"]:
                # Display each grasp on the original image, colored by
                # predicted success.
                # norm_q_values = ((q_values - np.min(q_values)) /
                #                   (np.max(q_values) - np.min(q_values)))
                norm_q_values = q_values
                title = "Sampled Grasps Iter %d" % (j)
                if self._vis_grasp_affordance_map:
                    self._plot_grasp_affordance_map(
                        state,
                        grasp_affordance_map,
                        grasps=grasps,
                        q_values=norm_q_values,
                        scale=2.0,
                        title=title,
                        save_fname="cem_iter_{}.png".format(j),
                        save_path=state_output_dir)
                display_grasps_and_q_values = zip(grasps, q_values)
                display_grasps_and_q_values = sorted(
                    display_grasps_and_q_values, key=lambda x: x[1])
                vis.figure(size=(GeneralConstants.FIGSIZE,
                                 GeneralConstants.FIGSIZE))
                vis.imshow(rgbd_im.depth,
                           vmin=self.config["vis"]["vmin"],
                           vmax=self.config["vis"]["vmax"])
                for grasp, q in display_grasps_and_q_values:
                    vis.grasp(grasp,
                              scale=2.0,
                              jaw_width=2.0,
                              show_center=False,
                              show_axis=True,
                              color=plt.cm.RdYlBu(q))
                vis.title("Sampled grasps iter %d" % (j))
                filename = None
                if self._logging_dir is not None:
                    filename = os.path.join(self._logging_dir,
                                            "cem_iter_%d.png" % (j))
                vis.show(filename)

            # Fit elite set.
            elite_start = time()
            num_refit = max(int(np.ceil(self._gmm_refit_p * num_grasps)), 1)
            elite_q_values = [i[0] for i in q_values_and_indices[:num_refit]]
            elite_grasp_indices = [
                i[1] for i in q_values_and_indices[:num_refit]
            ]
            elite_grasps = [grasps[i] for i in elite_grasp_indices]
            elite_grasp_arr = np.array([g.feature_vec for g in elite_grasps])

            if self.config["vis"]["elite_grasps"]:
                # Display each grasp on the original image, colored by
                # predicted success.
                norm_q_values = (elite_q_values - np.min(elite_q_values)) / (
                    np.max(elite_q_values) - np.min(elite_q_values))
                vis.figure(size=(GeneralConstants.FIGSIZE,
                                 GeneralConstants.FIGSIZE))
                vis.imshow(rgbd_im.depth,
                           vmin=self.config["vis"]["vmin"],
                           vmax=self.config["vis"]["vmax"])
                for grasp, q in zip(elite_grasps, norm_q_values):
                    vis.grasp(grasp,
                              scale=1.5,
                              show_center=False,
                              show_axis=True,
                              color=plt.cm.RdYlBu(q))
                vis.title("Elite grasps iter %d" % (j))
                filename = None
                if self._logging_dir is not None:
                    filename = os.path.join(self._logging_dir,
                                            "elite_set_iter_%d.png" % (j))
                vis.show(filename)

            # Normalize elite set.
            elite_grasp_mean = np.mean(elite_grasp_arr, axis=0)
            elite_grasp_std = np.std(elite_grasp_arr, axis=0)
            elite_grasp_std[elite_grasp_std == 0] = 1e-6
            elite_grasp_arr = (elite_grasp_arr -
                               elite_grasp_mean) / elite_grasp_std
            self._logger.info("Elite set computation took %.3f sec" %
                              (time() - elite_start))

            # Fit a GMM to the top samples.
            num_components = max(
                int(np.ceil(self._gmm_component_frac * num_refit)), 1)
            uniform_weights = (1.0 / num_components) * np.ones(num_components)
            gmm = GaussianMixture(n_components=num_components,
                                  weights_init=uniform_weights,
                                  reg_covar=self._gmm_reg_covar)
            train_start = time()
            gmm.fit(elite_grasp_arr)
            self._logger.info("GMM fitting with %d components took %.3f sec" %
                              (num_components, time() - train_start))

            # Sample the next grasps.
            grasps = []
            loop_start = time()
            num_tries = 0
            while (len(grasps) < self._num_gmm_samples
                   and num_tries < self._max_resamples_per_iteration):
                # Sample from GMM.
                sample_start = time()
                grasp_vecs, _ = gmm.sample(n_samples=self._num_gmm_samples)
                grasp_vecs = elite_grasp_std * grasp_vecs + elite_grasp_mean
                self._logger.info("GMM sampling took %.3f sec" %
                                  (time() - sample_start))

                # Convert features to grasps and store if in segmask.
                for k, grasp_vec in enumerate(grasp_vecs):
                    feature_start = time()
                    if grasp_type == "parallel_jaw":
                        # Form grasp object.
                        grasp = Grasp2D.from_feature_vec(
                            grasp_vec,
                            width=self._gripper_width,
                            camera_intr=camera_intr)
                    elif grasp_type == "suction":
                        # Read depth and approach axis.
                        u = int(min(max(grasp_vec[1], 0), depth_im.height - 1))
                        v = int(min(max(grasp_vec[0], 0), depth_im.width - 1))
                        grasp_depth = depth_im[u, v, 0]

                        # Approach axis.
                        grasp_axis = -normal_cloud_im[u, v]

                        # Form grasp object.
                        grasp = SuctionPoint2D.from_feature_vec(
                            grasp_vec,
                            camera_intr=camera_intr,
                            depth=grasp_depth,
                            axis=grasp_axis)
                    elif grasp_type == "multi_suction":
                        # Read depth and approach axis.
                        u = int(min(max(grasp_vec[1], 0), depth_im.height - 1))
                        v = int(min(max(grasp_vec[0], 0), depth_im.width - 1))
                        grasp_depth = depth_im[u, v]

                        # Approach_axis.
                        grasp_axis = -normal_cloud_im[u, v]

                        # Form grasp object.
                        grasp = MultiSuctionPoint2D.from_feature_vec(
                            grasp_vec,
                            camera_intr=camera_intr,
                            depth=grasp_depth,
                            axis=grasp_axis)
                    self._logger.debug("Feature vec took %.5f sec" %
                                       (time() - feature_start))

                    bounds_start = time()
                    # Check in bounds.
                    if (state.segmask is None or
                        (grasp.center.y >= 0
                         and grasp.center.y < state.segmask.height
                         and grasp.center.x >= 0
                         and grasp.center.x < state.segmask.width
                         and np.any(state.segmask[int(grasp.center.y),
                                                  int(grasp.center.x)] != 0)
                         and grasp.approach_angle < self._max_approach_angle)
                            and (self._grasp_constraint_fn is None
                                 or self._grasp_constraint_fn(grasp))):

                        # Check validity according to filters.
                        grasps.append(grasp)
                    self._logger.debug("Bounds took %.5f sec" %
                                       (time() - bounds_start))
                    num_tries += 1

            # Check num grasps.
            num_grasps = len(grasps)
            if num_grasps == 0:
                self._logger.warning("No valid grasps could be found")
                raise NoValidGraspsException()
            self._logger.info("Resample loop took %.3f sec" %
                              (time() - loop_start))
            self._logger.info("Resampling took %.3f sec" %
                              (time() - resample_start))

        # Predict final set of grasps.
        predict_start = time()
        q_values = self._grasp_quality_fn(state, grasps, params=self._config)
        self._logger.info("Final prediction took %.3f sec" %
                          (time() - predict_start))

        if self.config["vis"]["grasp_candidates"]:
            # Display each grasp on the original image, colored by predicted
            # success.
            # norm_q_values = ((q_values - np.min(q_values)) /
            #                   (np.max(q_values) - np.min(q_values)))
            norm_q_values = q_values
            title = "Final Sampled Grasps"
            if self._vis_grasp_affordance_map:
                self._plot_grasp_affordance_map(
                    state,
                    grasp_affordance_map,
                    grasps=grasps,
                    q_values=norm_q_values,
                    scale=2.0,
                    title=title,
                    save_fname="final_sampled_grasps.png",
                    save_path=state_output_dir)
            display_grasps_and_q_values = zip(grasps, q_values)
            display_grasps_and_q_values = sorted(display_grasps_and_q_values,
                                                 key=lambda x: x[1])
            vis.figure(size=(GeneralConstants.FIGSIZE,
                             GeneralConstants.FIGSIZE))
            vis.imshow(rgbd_im.depth,
                       vmin=self.config["vis"]["vmin"],
                       vmax=self.config["vis"]["vmax"])
            for grasp, q in display_grasps_and_q_values:
                vis.grasp(grasp,
                          scale=2.0,
                          jaw_width=2.0,
                          show_center=False,
                          show_axis=True,
                          color=plt.cm.RdYlBu(q))
            vis.title("Sampled grasps iter %d" % (j))
            filename = None
            if self._logging_dir is not None:
                filename = os.path.join(self._logging_dir,
                                        "cem_iter_%d.png" % (j))
            vis.show(filename)

        return grasps, q_values

    def _action(self, state):
        """Plans the grasp with the highest probability of success on
        the given RGB-D image.

        Attributes
        ----------
        state : :obj:`RgbdImageState`
            Image to plan grasps on.

        Returns
        -------
        :obj:`GraspAction`
            Grasp to execute.
        """
        # Parse state.
        rgbd_im = state.rgbd_im
        depth_im = rgbd_im.depth
        #        camera_intr = state.camera_intr
        #        segmask = state.segmask

        # Plan grasps.
        grasps, q_values = self.action_set(state)

        # Select grasp.
        index = self.select(grasps, q_values)
        grasp = grasps[index]
        q_value = q_values[index]
        if self.config["vis"]["grasp_plan"]:
            title = "Best Grasp: d=%.3f, q=%.3f" % (grasp.depth, q_value)
            vis.figure()
            vis.imshow(depth_im,
                       vmin=self.config["vis"]["vmin"],
                       vmax=self.config["vis"]["vmax"])
            vis.grasp(grasp,
                      scale=5.0,
                      show_center=False,
                      show_axis=True,
                      jaw_width=1.0,
                      grasp_axis_width=0.2)
            vis.title(title)
            filename = None
            if self._logging_dir is not None:
                filename = os.path.join(self._logging_dir, "planned_grasp.png")
            vis.show(filename)

        # Form return image.
        image = depth_im
        if isinstance(self._grasp_quality_fn, GQCnnQualityFunction):
            image_arr, _ = self._grasp_quality_fn.grasps_to_tensors([grasp],
                                                                    state)
            image = DepthImage(image_arr[0, ...], frame=rgbd_im.frame)

        # Return action.
        action = GraspAction(grasp, q_value, image)
        return action
    