import abc
import warnings

import numpy as np
import torch
from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING
from mmcv.runner import BaseModule


class Encoding(BaseModule, abc.ABC):
    """
    Abstract base class for all encodings
    """

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg=init_cfg)

    def _apply_mask(self, encoding, mask):
        """Utility to apply the img pos mask

        Parameters
        ----------
        encoding : torch.Tensor
            Encoding to apply mask to, shape (cams x bs x dims x h x w)
        mask : torch.BoolTensor
            (bs, h,w), padding is performed in lower right corner.
        """
        # apply mask to mask invalid values
        # add feats dim
        mask = torch.unsqueeze(mask, dim=1)
        # add cam_num dim
        mask = torch.unsqueeze(mask, dim=0)

        # mask is 0 for valid and 1 for invalid (bool)
        # -> invert then convert to float and multiply
        encoding = encoding * (~mask).float()
        return encoding

    @abc.abstractproperty
    def embed_dims(self):
        """Property for embedding dimensionality

        Returns
        -------
        int
            Dimensions of encoding.
        """
        return NotImplementedError

    @abc.abstractmethod
    def forward(self, mask, img_metas=None):
        """Abstract forward function for camera embeddings


        Parameters
        ----------
        mask : torch.BoolTensor
            (bs, h,w), padding is performed in lower right corner.
        img_metas : dict, optional
            Img meta containing calibration information, by default None

        Raises
        ------
        NotImplementedError
            Abstract method
        """

        raise NotImplementedError


@POSITIONAL_ENCODING.register_module()
class FixedGeometricEncoding(Encoding):
    """Geometric positional encoding that computes direction rays per pixel
    """

    def __init__(self, apply_global_rot=True, init_cfg=None):
        """Creates the FixedGeometricEncoding

        Parameters
        ----------
        apply_global_rot : bool, optional
            Whether to apply global rotation (wrt. to reference) or not, by default True
        init_cfg : dict, optional
            mmcv init config, by default None
        """
        super(FixedGeometricEncoding, self).__init__(init_cfg)
        self._apply_global_rot = apply_global_rot

    def forward(self, mask, img_metas):
        """@See base class for details.
        """
        BS, H, W = mask.shape
        NUM_CAMS = len(img_metas[0]["cam_intrinsic"])

        feats_shape = mask.shape[1:]
        # TODO refactor: we assume same img shape for entire batch
        img_shape = img_metas[0]["img_shape"]

        # xs and ys do include "invalid" values at idxs that correspond to padded pixels
        xs, ys = torch.meshgrid(
            torch.arange(
                0, feats_shape[1], device=mask.device, requires_grad=False),
            torch.arange(
                0, feats_shape[0], device=mask.device, requires_grad=False),
        )
        xs = xs.float()
        ys = ys.float()

        # 3 x width x height
        cam_embeddings = torch.cat(
            [
                xs.unsqueeze(dim=0),
                ys.unsqueeze(dim=0),
                torch.zeros((1, xs.shape[0], xs.shape[1]),
                            device=xs.device, requires_grad=False),
            ],
            axis=0,
        )

        # 3 x w x h -> h x w x 3
        cam_embeddings = cam_embeddings.permute((2, 1, 0))
        # add cam and batch dim
        # h x w x 3 -> cams x bs x h x w x 3
        # clone to assure that each cam has its own storage (expand only creates references)
        cam_embeddings = cam_embeddings.expand(
            (
                NUM_CAMS,
                BS,
            )
            + cam_embeddings.shape
        ).clone()

        for cam_idx in range(len(img_shape)):
            for s_id in range(BS):
                # TODO refactor as initial check / caching
                # allow for different scales via scale_mat @ K
                full_img_shape = img_shape[cam_idx]
                feature_scale_x = W / full_img_shape[1]
                feature_scale_y = H / full_img_shape[0]

                if not np.allclose(feature_scale_x, feature_scale_y):
                    # the feature scale was not the same (due to uneven division)
                    # padding fixes this -> one side is too long -> too high scale
                    # to fix we use the smaller feature scale
                    warnings.warn(
                        "x/y feature scale diff, double check padding...")
                    feature_scale = min(feature_scale_x, feature_scale_y)

                else:
                    feature_scale = feature_scale_x

                K = img_metas[s_id]["cam_intrinsic"][cam_idx]

                # TODO refactor: assumes fx == fy
                # scaling can be accounted for by simply scaling focal length and principal point
                scale_factor = img_metas[s_id]["scale_factor"]
                scale_factor = scale_factor * feature_scale
                cx = K[0][2]
                cy = K[1][2]
                focal_length = K[0][0]

                # cx
                cam_embeddings[cam_idx, s_id, :, :, 0] -= cx * scale_factor
                # cy
                cam_embeddings[cam_idx, s_id, :, :, 1] -= cy * scale_factor
                # focal length
                cam_embeddings[cam_idx, s_id, :, :,
                               2] = focal_length * scale_factor

                # apply rotation wrt to reference system
                cam_T_lidar = torch.tensor(
                    img_metas[s_id]["lidar2cam"][cam_idx], device=mask.device, dtype=torch.float, requires_grad=False
                )
                # TODO refactoring: use caching if extrinsics did not change
                lidar_T_cam = torch.inverse(cam_T_lidar)

                # fast way for lidar_r_cam @ cam_pixels
                ori_shape = cam_embeddings[cam_idx][s_id].shape
                points = torch.reshape(cam_embeddings[cam_idx][s_id], (-1, 3))

                # normalize
                points = points / torch.norm(points, dim=1, keepdim=True)

                if self._apply_global_rot:
                    points = points @ lidar_T_cam[0:3, 0:3].T

                # # for debug:
                # ones = torch.ones((len(points), 1), device=points.device)
                # points = torch.cat(
                #     [points, ones], dim=-1)
                # # full transformation (requires homogenous coordinates)
                # points = points @ lidar_T_cam.T
                # points = points[:, 0:3]

                cam_embeddings[cam_idx][s_id] = torch.reshape(
                    points, ori_shape)

        # cams x bs x h x w x dims -> cams x bs x dims x h x w
        cam_embeddings = torch.permute(cam_embeddings, (0, 1, 4, 2, 3))

        # apply mask to mask invalid values
        cam_embeddings = self._apply_mask(cam_embeddings, mask)

        return cam_embeddings

    @ property
    def embed_dims(self):
        return 3  # 3d encding (direction vectors)

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f"(num_feats=3, "
        return repr_str
