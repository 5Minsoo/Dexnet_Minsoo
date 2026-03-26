import numpy as np
import scipy.ndimage as ndimage


class RealsenseDepthNoise:
    """Simulates Intel RealSense depth sensor noise on synthetic depth images.

    Applies:
    1. Gaussian noise proportional to depth^2 (RealSense stereo noise model)
    2. Random pixel dropout (scattered black dots on object surface)
    3. Edge erosion (missing/corrupted depth at object boundaries)
    4. Thin structure dropout (narrow regions lose depth readings)

    Parameters
    ----------
    gaussian_scale : float
        Scale factor for depth-dependent Gaussian noise. σ = gaussian_scale * z^2.
        Default 0.002 corresponds to ~2mm noise at 1m depth.
    dropout_rate : float
        Base probability of a valid pixel dropping to 0. Default 0.005.
    edge_erosion_width : int
        Width in pixels of the edge region to corrupt. Default 2.
    edge_dropout_rate : float
        Probability of dropping a pixel in the edge region. Default 0.5.
    thin_threshold : int
        Minimum width in pixels for a structure to be retained.
        Regions thinner than this get dropped. Default 3.
    thin_dropout_rate : float
        Probability of dropping pixels in thin regions. Default 0.8.
    """

    def __init__(self,
                 gaussian_scale=0.002,
                 dropout_rate=0.005,
                 edge_erosion_width=2,
                 edge_dropout_rate=0.5,
                 thin_threshold=3,
                 thin_dropout_rate=0.8):
        self.gaussian_scale = gaussian_scale
        self.dropout_rate = dropout_rate
        self.edge_erosion_width = edge_erosion_width
        self.edge_dropout_rate = edge_dropout_rate
        self.thin_threshold = thin_threshold
        self.thin_dropout_rate = thin_dropout_rate

    def apply(self, depth_im):
        """Apply all noise effects to a depth image.

        Parameters
        ----------
        depth_im : (H, W) np.ndarray, float32
            Clean synthetic depth image. 0 = no reading (background).

        Returns
        -------
        (H, W) np.ndarray, float32
            Noisy depth image.
        """
        noisy = depth_im.copy()
        obj_mask = depth_im > 0

        noisy = self._add_gaussian_noise(noisy, obj_mask)
        noisy = self._add_random_dropout(noisy, obj_mask)
        noisy = self._add_edge_erosion(noisy, obj_mask)
        noisy = self._add_thin_structure_dropout(noisy, obj_mask)

        noisy[noisy < 0] = 0
        return noisy

    def _add_gaussian_noise(self, depth_im, obj_mask):
        """σ = gaussian_scale * z^2  (RealSense stereo depth noise model)"""
        sigma = self.gaussian_scale * depth_im ** 2
        noise = np.random.randn(*depth_im.shape).astype(np.float32) * sigma
        depth_im[obj_mask] += noise[obj_mask]
        return depth_im

    def _add_random_dropout(self, depth_im, obj_mask):
        """Randomly set valid pixels to 0 (sensor misses)."""
        drop = np.random.rand(*depth_im.shape) < self.dropout_rate
        depth_im[obj_mask & drop] = 0
        return depth_im

    def _add_edge_erosion(self, depth_im, obj_mask):
        """Corrupt pixels near depth discontinuities (object edges)."""
        # find edge pixels: dilate the background mask, the overlap with
        # the object mask gives the edge band on the object side
        bg_mask = ~obj_mask
        dilated_bg = ndimage.binary_dilation(
            bg_mask, iterations=self.edge_erosion_width
        )
        edge_band = obj_mask & dilated_bg

        # also detect internal depth discontinuities (sharp depth jumps)
        grad_x = np.abs(np.diff(depth_im, axis=1, prepend=0))
        grad_y = np.abs(np.diff(depth_im, axis=0, prepend=0))
        depth_grad = np.maximum(grad_x, grad_y)
        # threshold: jump > 1cm counts as a discontinuity
        discontinuity = (depth_grad > 0.01) & obj_mask
        discontinuity_band = ndimage.binary_dilation(
            discontinuity, iterations=self.edge_erosion_width
        )

        corrupt_region = edge_band | discontinuity_band
        drop = np.random.rand(*depth_im.shape) < self.edge_dropout_rate
        depth_im[corrupt_region & drop] = 0
        return depth_im

    def _add_thin_structure_dropout(self, depth_im, obj_mask):
        """Drop depth in regions where the object is thinner than thin_threshold pixels."""
        # compute local thickness via distance transform from background
        dist = ndimage.distance_transform_edt(obj_mask)
        thin_mask = (dist > 0) & (dist < self.thin_threshold)

        drop = np.random.rand(*depth_im.shape) < self.thin_dropout_rate
        depth_im[thin_mask & drop] = 0
        return depth_im
