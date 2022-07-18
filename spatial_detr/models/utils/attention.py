import torch.nn.functional as F
import math
import warnings
from typing import Optional, Tuple

import torch.nn as nn
import torch
import numpy as np
from mmcv.cnn.bricks.transformer import ATTENTION
from mmcv.runner import BaseModule

from mmcv import ConfigDict, deprecated_api_warning
from mmcv.cnn import Linear, build_activation_layer, build_norm_layer
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn import xavier_init, constant_init

from spatial_detr.thirdparty.detr3d.util import inverse_sigmoid

Tensor = torch.Tensor

from spatial_detr.thirdparty.torch.custom_multi_head_attention import CustomMultiheadAttention


@ATTENTION.register_module()
class QueryValueProjectCrossAttention(BaseModule):
    """A cross-attention module that uses sensor-relative queries and global value projection"""

    def __init__(self,
                 embed_dims,
                 num_heads,
                 pc_range=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        """Constructs the Cross Attention module

        Parameters
        ----------
        embed_dims : int
            latent dimensionality
        num_heads : int
            number of heads for multi-head attention
        pc_range : list, optional
            point cloud range for coorinate normalization, by default None
        attn_drop : float, optional
            dropout probability in attention, by default 0.
        proj_drop : float, optional
            output dropout probability, by default 0.
        dropout_layer : dict, optional
            Configuration of dropout layer, by default dict(type='Dropout', drop_prob=0.)
        init_cfg : dict, optional
            Initialization configuration, by default None
        batch_first : bool, optional
            Batch mode, currently only batch_first=False is fully supported, by default False

        Returns
        -------
        tensor
            Update query vectors
        """
        super(QueryValueProjectCrossAttention, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn('The arguments `dropout` in MultiheadAttention '
                          'has been deprecated, now you can separately '
                          'set `attn_drop`(float), proj_drop(float), '
                          'and `dropout_layer`(dict) ')
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.pc_range = pc_range

        # query 3d -> latent
        self.query_loc2latent = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        # value -> depth
        self.value2depth = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.Linear(self.embed_dims, 1),
            nn.ReLU(inplace=True),
        )
        # value projection: 3d -> latent
        self.value_loc2latent = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        # additional position feature 3d -> latent (for query)
        self.position_encoder_out = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        # custom attn function
        self.attn = CustomMultiheadAttention(embed_dims, num_heads, attn_drop,
                                             **kwargs)
        if self.batch_first:
            raise NotImplementedError("batch first mode is not yet supported")

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    def _attn_weights_only_dot_prod_attn(self,
                                         q: Tensor,
                                         k: Tensor,
                                         v: Tensor,
                                         attn_mask: Optional[Tensor] = None,
                                         dropout_p: float = 0.0,
                                         **kwargs) -> Tuple[Tensor, Tensor]:
        """Spatial attention module

        Parameters
        ----------
        q : Tensor (Cams x Batch x Queries x dims)
            object queries (spatially influenced)
        k : Tensor (Cams x Batch x Keys x dims)
            keys (spatially influenced)
        v : Tensor (Cams x Batch x Values x dims)
            _description_
        attn_mask : Optional[Tensor], optional
            additional mask for attention, by default None (currently not supported)
        dropout_p : float, optional
            drop out probability, by default 0.0

        Returns
        -------
        Tuple[Tensor, Tensor]
            weighted values and attention weights
        """

        CAMS, B, Q, E = q.shape

        assert attn_mask is None, "attn_mask is not supported currently"

        # normalize before stacking to not rescale the direction vector
        q = q / math.sqrt(E)
        k = k / math.sqrt(E)

        patches_per_img = k.shape[2]

        attn_full = torch.empty(
            (CAMS, B, Q, patches_per_img), device=q.device)

        for cam_idx in range(CAMS):
            # perform the attention for each query in each camera
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            attn = torch.bmm(q[cam_idx], k[cam_idx].transpose(-2, -1))

            attn = F.softmax(attn, dim=-1)
            if dropout_p > 0.0:
                attn = F.dropout(attn, p=dropout_p)

            attn_full[cam_idx] = attn

        # cams x b x patches x dims -> b x cams x patches x dims
        v = v.permute(1, 0, 2, 3)
        v = v.reshape(B, -1, E)

        # cams x B x Nt x patches -> B x Nt x cams x patches
        attn_full = attn_full.permute(1, 2, 0, 3)
        attn_full = attn_full.reshape(B, Q, -1)

        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(attn_full, v)
        return output, attn_full

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                reference_points=None,
                img_metas=None,
                reg_branches=None,
                **kwargs):
        """
        """
        if identity is None:
            identity = query

        if len(key_pos.shape) == 4:
            # multi scale case (is h*w already)
            key_pos = key_pos.permute(1, 3, 0, 2)
            value = value.permute(1, 3, 0, 2)
        else:

            # input shape: bs x cam x dims x h x w
            # permute -> cams x h x w x bs  x dims
            # -> cams x h * w x bs x dim
            key_pos = key_pos.permute(1, 3, 4, 0, 2)
            key_pos = key_pos.reshape(
                key_pos.shape[0], -1, key_pos.shape[3], key_pos.shape[4])

            value = value.permute(1, 3, 4, 0, 2)
            value = value.reshape(
                value.shape[0], -1, value.shape[3], value.shape[4])

        # feats (vals and keys) as well as encoding come in shape
        # cams x patches x bs x dims
        CAMS, PATCHES_PER_CAM, BS = value.shape[0:3]
        QUERIES = reference_points.shape[1]

        # TODO refactor to avoid the loop
        # bs x cams x 4 x 4 contains transforms for each batch and each cam
        cam_T_lidar_tensor = torch.empty(
            (BS, CAMS, 4, 4), device=value.device, requires_grad=False)
        lidar_T_cam_tensor = torch.empty(
            (BS, CAMS, 4, 4), device=value.device, requires_grad=False)

        for i, meta in enumerate(img_metas):  # over batch dim
            cam_T_lidar_cams = torch.tensor(np.stack(
                meta['lidar2cam']), dtype=torch.float, device=value.device, requires_grad=False)
            cam_T_lidar_tensor[i] = cam_T_lidar_cams

            for cam_idx in range(CAMS):
                lidar_T_cam_tensor[i][cam_idx] = torch.inverse(
                    cam_T_lidar_cams[cam_idx])
        # for multiplication use (A*B) == (B^T * A^T)
        # bs x cams x 4_1, 4_2 -> cams x bs x 4_2 x 4_1 (transpose / permute)
        cam_T_lidar_tensor = cam_T_lidar_tensor.permute(1, 0, 3, 2)
        lidar_T_cam_tensor = lidar_T_cam_tensor.permute(1, 0, 3, 2)

        # reference points from sigmoid space to cartesian:
        reference_points_orig = reference_points.clone()
        # Due to the inplace modificiations a clone is needed (autograd anomaly)
        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

        # The ones are just appended but do not need a gradient flow
        ones_query = torch.ones(
            (BS, QUERIES, 1), device=reference_points.device, requires_grad=False)

        reference_points_homogenous = torch.cat(
            [reference_points, ones_query], dim=-1)

        query_per_cam = torch.empty(
            (CAMS, QUERIES, BS, self.embed_dims), device=value.device)

        # create latent pos encoding for keys
        key_pos_feature = self.query_loc2latent(key_pos)

        # the values will also get a pos feature
        # to save memory feats_with_dir are also the keys
        feats_with_dir = value + key_pos_feature

        # get a single depth value
        value_3d_cam = self.value2depth(feats_with_dir)
        # compute full 3d vector
        value_3d_cam = value_3d_cam * key_pos

        # for bmm the batch dimension needs to be first -> permute
        # cams x patches x bs x 3 -> cams x bs x patches x 3
        value_3d_cam = value_3d_cam.permute(0, 2, 1, 3)
        ones_value = torch.ones((BS, PATCHES_PER_CAM, 1),
                                device=value.device, requires_grad=False)

        # since inplace assignments to values produces torch gradient anomalies we need a separate storage container
        values_global = torch.empty_like(value)

        # for each cam project query ref points
        for cam_idx in range(CAMS):
            # projections might be different for each element in a batch
            # apply to entire batch
            # remove homogenous dimension
            reference_points_cam = torch.bmm(
                reference_points_homogenous, cam_T_lidar_tensor[cam_idx])[..., 0:3]

            # normalize -> direction vector instead of point in 3d space
            reference_points_cam = reference_points_cam / \
                torch.norm(reference_points_cam, dim=-1, keepdim=True)

            query_position_feature = self.query_loc2latent(
                reference_points_cam)
            # bs x queries x dims -> queries x bs x dims
            query_position_feature = query_position_feature.permute(
                1, 0, 2)

            # query = query + query encoding + position feature (relative to cam)
            query_per_cam[cam_idx] = query + \
                query_pos + query_position_feature

            value_3d_cam_current = torch.cat(
                [value_3d_cam[cam_idx], ones_value], dim=-1)

            value_3d_ref = torch.bmm(
                value_3d_cam_current, lidar_T_cam_tensor[cam_idx])[..., 0:3]

            # value global feature -> latent
            value_3d_ref = self.value_loc2latent(value_3d_ref)

            # element wise add feature to value
            # bs x patches x latent -> patches x bs x latent
            value_3d_ref = value_3d_ref.permute(1, 0, 2)
            values_global[cam_idx] = value[cam_idx] + value_3d_ref

        # the keys and queries are camera specific now, values are in global coordinates
        weighted_values, _ = self.attn(
            query=query_per_cam,
            key=feats_with_dir,
            value=values_global,
            key_pos=key_pos,
            query_pos=None,
            attn_mask=None,
            key_padding_mask=None,
            need_weights=False,
            attn_func=self._attn_weights_only_dot_prod_attn
        )

        pos_feat = self.position_encoder_out(
            inverse_sigmoid(reference_points_orig)).permute(1, 0, 2)

        return identity + self.dropout_layer(self.proj_drop(weighted_values)) + pos_feat


@ATTENTION.register_module()
class QueryCenterValueProjectCrossAttention(QueryValueProjectCrossAttention):
    """A cross attention module that forces query centers to be in front of the camera plane and uses global value projection"""

    def _attn_weights_query_center_only_dot_prod_attn(self,
                                                      q: Tensor,
                                                      k: Tensor,
                                                      v: Tensor,
                                                      query_masks=None,
                                                      attn_mask: Optional[Tensor] = None,
                                                      dropout_p: float = 0.0,
                                                      **kwargs) -> Tuple[Tensor, Tensor]:
        """Spatial attention module with a geometric constraint that the object center needs to be in front of the camera

        Parameters
        ----------
        q : Tensor (Cams x Batch x Queries x dims)
            object queries (spatially influenced)
        k : Tensor (Cams x Batch x Keys x dims)
            keys (spatially influenced)
        v : Tensor (Cams x Batch x Values x dims)
            _description_
        attn_mask : Optional[Tensor], optional
            additional mask for attention, by default None (currently not supported)
        dropout_p : float, optional
            drop out probability, by default 0.0

        Returns
        -------
        Tuple[Tensor, Tensor]
            weighted values and attention weights
        """

        CAMS, B, Q, E = q.shape
        _, _, Ns, E = v.shape

        # c x q x bs x 1 -> c x bs x q x 1 (bs != head *bs)
        query_masks = query_masks.permute(0, 2, 1, 3)
        heads = int(B / query_masks.shape[1])
        # c x bs x q x 1 -> c x bs * heads x q x 1
        query_masks = query_masks.repeat(1, heads, 1, 1)

        # normalize before stacking to not rescale the direction vector
        q = q / math.sqrt(E)
        k = k / math.sqrt(E)

        patches_per_img = k.shape[2]

        attn_full = torch.empty(
            (CAMS, B, Q, patches_per_img), device=q.device)

        for cam_idx in range(CAMS):
            # perform the attention for each query in each camera
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            attn = torch.bmm(q[cam_idx], k[cam_idx].transpose(-2, -1))

            attn = F.softmax(attn, dim=-1)
            if dropout_p > 0.0:
                attn = F.dropout(attn, p=dropout_p)

            # set weights for all non visible query centers to zero
            attn = attn * query_masks[cam_idx]

            attn_full[cam_idx] = attn

        # cams x b x patches x dims -> b x cams x patches x dims
        v = v.permute(1, 0, 2, 3)
        v = v.reshape(B, -1, E)

        # cams x B x Nt x patches -> B x Nt x cams x patches
        attn_full = attn_full.permute(1, 2, 0, 3)
        attn_full = attn_full.reshape(B, Q, -1)

        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(attn_full, v)
        return output, attn_full

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                reference_points=None,
                img_metas=None,
                reg_branches=None,
                **kwargs):
        """@See base for details.
        """

        if identity is None:
            identity = query

        if len(key_pos.shape) == 4:
            # multi scale (is h*w already)
            key_pos = key_pos.permute(1, 3, 0, 2)
            value = value.permute(1, 3, 0, 2)
            key = key.permute(1, 3, 0, 2)

        else:
            # input shape: bs x cam x dims x h x w
            # permute -> cams x h x w x bs  x dims
            # -> cams x h * w x bs x dim
            key_pos = key_pos.permute(1, 3, 4, 0, 2)
            key_pos = key_pos.reshape(
                key_pos.shape[0], -1, key_pos.shape[3], key_pos.shape[4])

            value = value.permute(1, 3, 4, 0, 2)
            value = value.reshape(
                value.shape[0], -1, value.shape[3], value.shape[4])

            key = key.permute(1, 3, 4, 0, 2)
            key = key.reshape(key.shape[0], -1, key.shape[3], key.shape[4])

        # feats (vals and keys) as well as encoding come in shape
        # cams x patches x bs x dims
        CAMS, PATCHES_PER_CAM, BS = value.shape[0:3]
        QUERIES = reference_points.shape[1]

        # TODO refactor to avoid the loop
        # bs x cams x 4 x 4 contains transforms for each batch and each cam
        cam_T_lidar_tensor = torch.empty(
            (BS, CAMS, 4, 4), device=value.device, requires_grad=False)
        lidar_T_cam_tensor = torch.empty(
            (BS, CAMS, 4, 4), device=value.device, requires_grad=False)

        for i, meta in enumerate(img_metas):  # over batch dim
            cam_T_lidar_cams = torch.tensor(np.stack(
                meta['lidar2cam']), dtype=torch.float, device=value.device, requires_grad=False)
            cam_T_lidar_tensor[i] = cam_T_lidar_cams

            for cam_idx in range(CAMS):
                lidar_T_cam_tensor[i][cam_idx] = torch.inverse(
                    cam_T_lidar_cams[cam_idx])
        # for multiplication use (A*B) == (B^T * A^T)
        # bs x cams x 4_1, 4_2 -> cams x bs x 4_2 x 4_1 (transpose / permute)
        cam_T_lidar_tensor = cam_T_lidar_tensor.permute(1, 0, 3, 2)
        lidar_T_cam_tensor = lidar_T_cam_tensor.permute(1, 0, 3, 2)

        # reference points from sigmoid space to cartesian:
        reference_points_orig = reference_points.clone()
        # Due to the inplace modificiations a clone is needed (autograd anomaly)
        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

        # The ones are just appended but do not need a gradient flow
        ones_query = torch.ones(
            (BS, QUERIES, 1), device=reference_points.device, requires_grad=False)

        reference_points_homogenous = torch.cat(
            [reference_points, ones_query], dim=-1)

        query_per_cam = torch.empty(
            (CAMS, QUERIES, BS, self.embed_dims), device=value.device)

        # create latent pos encoding for keys
        key_pos_feature = self.query_loc2latent(key_pos)

        # the values will also get a pos feature
        # to save memory feats_with_dir are also the keys
        value_depth = self.value2depth(value + key_pos_feature)
        value_3d_cam = value_depth * key_pos

        # for bmm the batch dimension needs to be first -> permute
        # cams x patches x bs x 3 -> cams x bs x patches x 3
        value_3d_cam = value_3d_cam.permute(0, 2, 1, 3)
        ones_value = torch.ones((BS, PATCHES_PER_CAM, 1),
                                device=value.device, requires_grad=False)

        # since inplace assignments to values produces torch gradient anomalies we need a separate storage container
        values_global = torch.empty_like(value)

        # for each cam project query ref points

        cam_front_facing = torch.tensor(
            [0, 0, 1], dtype=torch.float, device=value.device, requires_grad=False)

        cam_front_facing = cam_front_facing.repeat(BS, 1)
        # add dim at end
        cam_front_facing = cam_front_facing.unsqueeze(-1)

        query_masks = torch.empty(
            (CAMS, QUERIES, BS, 1), device=query.device, requires_grad=False, dtype=torch.bool)

        for cam_idx in range(CAMS):
            # projections might be different for each element in a batch
            # apply to entire batch
            # remove homogenous dimension
            reference_points_cam = torch.bmm(
                reference_points_homogenous, cam_T_lidar_tensor[cam_idx])[..., 0:3]

            # normalize -> direction vector instead of point in 3d space
            reference_points_cam = reference_points_cam / \
                torch.norm(reference_points_cam, dim=-1, keepdim=True)

            scalar_prod_query_cam = torch.bmm(
                reference_points_cam, cam_front_facing)
            # bs x q x 1 -> q x bs x 1
            scalar_prod_query_cam = scalar_prod_query_cam.permute(1, 0, 2)
            query_masks[cam_idx] = scalar_prod_query_cam >= 0.0

            query_position_feature = self.query_loc2latent(
                reference_points_cam)
            # bs x queries x dims -> queries x bs x dims
            query_position_feature = query_position_feature.permute(
                1, 0, 2)

            # query = query + query encoding + position feature (relative to cam)
            query_per_cam[cam_idx] = query + \
                query_pos + query_position_feature

            value_3d_cam_current = torch.cat(
                [value_3d_cam[cam_idx], ones_value], dim=-1)

            value_3d_ref = torch.bmm(
                value_3d_cam_current, lidar_T_cam_tensor[cam_idx])[..., 0:3]

            # value global feature -> latent
            value_3d_ref = self.value_loc2latent(value_3d_ref)

            # element wise add feature to value
            # bs x patches x latent -> patches x bs x latent
            value_3d_ref = value_3d_ref.permute(1, 0, 2)
            values_global[cam_idx] = value[cam_idx] + value_3d_ref

        key = key + key_pos_feature

        # the keys and queries are camera specific now, values are in global coordinates
        weighted_values, _ = self.attn(
            query=query_per_cam,
            key=key,
            value=values_global,
            key_pos=key_pos,
            query_pos=None,
            query_masks=query_masks,
            attn_mask=None,
            key_padding_mask=None,
            need_weights=False,
            attn_func=self._attn_weights_query_center_only_dot_prod_attn
        )

        # TODO refactor different versions of reference_points
        pos_feat = self.position_encoder_out(
            inverse_sigmoid(reference_points_orig)).permute(1, 0, 2)

        return identity + self.dropout_layer(self.proj_drop(weighted_values)) + pos_feat


@ATTENTION.register_module()
class QueryOnlyProjectCrossAttention(QueryValueProjectCrossAttention):
    """A cross attention module that uses sensor-relative queries but no value projection"""

    def __init__(self,
                 embed_dims,
                 num_heads,
                 pc_range=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        """@See base class"""
        # dont initialize parent (dont initialize unwanted layers / modules)
        BaseModule.__init__(self, init_cfg)


        if 'dropout' in kwargs:
            warnings.warn('The arguments `dropout` in MultiheadAttention '
                          'has been deprecated, now you can separately '
                          'set `attn_drop`(float), proj_drop(float), '
                          'and `dropout_layer`(dict) ')
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.pc_range = pc_range

        # query 3d -> latent
        self.query_loc2latent = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        # additional 3d center feature -> latent
        self.position_encoder_out = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.attn = CustomMultiheadAttention(embed_dims, num_heads, attn_drop,
                                             **kwargs)
        if self.batch_first:
            raise NotImplementedError("batch first mode is not yet supported")

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    @ deprecated_api_warning({'residual': 'identity'},
                             cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                reference_points=None,
                img_metas=None,
                reg_branches=None,
                **kwargs):
        """
        @See base class for details
        """

        if identity is None:
            identity = query

        if len(key_pos.shape) == 4:
            # multi scale case (is h*w already)
            key_pos = key_pos.permute(1, 3, 0, 2)
            value = value.permute(1, 3, 0, 2)

        else:
            # input shape: bs x cam x dims x h x w
            # permute -> cams x h x w x bs  x dims
            # -> cams x h * w x bs x dim
            key_pos = key_pos.permute(1, 3, 4, 0, 2)
            key_pos = key_pos.reshape(
                key_pos.shape[0], -1, key_pos.shape[3], key_pos.shape[4])

            value = value.permute(1, 3, 4, 0, 2)
            value = value.reshape(
                value.shape[0], -1, value.shape[3], value.shape[4])

        # feats (vals and keys) as well as encoding come in shape
        # cams x patches x bs x dims
        CAMS, PATCHES_PER_CAM, BS = value.shape[0:3]
        QUERIES = reference_points.shape[1]

        # TODO refactor to avoid the loop
        # bs x cams x 4 x 4 contains transforms for each batch and each cam
        cam_T_lidar_tensor = torch.empty(
            (BS, CAMS, 4, 4), device=value.device, requires_grad=False)

        for i, meta in enumerate(img_metas):  # over batch dim
            cam_T_lidar_cams = torch.tensor(np.stack(
                meta['lidar2cam']), dtype=torch.float, device=value.device, requires_grad=False)
            cam_T_lidar_tensor[i] = cam_T_lidar_cams

        # for multiplication use (A*B) == (B^T * A^T)
        # bs x cams x 4_1, 4_2 -> cams x bs x 4_2 x 4_1 (transpose / permute)
        cam_T_lidar_tensor = cam_T_lidar_tensor.permute(1, 0, 3, 2)

        # reference points from sigmoid space to cartesian:
        reference_points_orig = reference_points.clone()
        # Due to the inplace modificiations a clone is needed (autograd anomaly)
        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

        # The ones are just appended but do not need a gradient flow
        ones_query = torch.ones(
            (BS, QUERIES, 1), device=reference_points.device, requires_grad=False)

        reference_points_homogenous = torch.cat(
            [reference_points, ones_query], dim=-1)

        query_per_cam = torch.empty(
            (CAMS, QUERIES, BS, self.embed_dims), device=value.device)

        # create latent pos encoding for keys
        key_pos_feature = self.query_loc2latent(key_pos)

        # the values will also get a pos feature
        # to save memory feats_with_dir are also the keys
        feats_with_dir = value + key_pos_feature

        # for each cam project query ref points
        for cam_idx in range(CAMS):
            # projections might be different for each element in a batch
            # apply to entire batch
            # remove homogenous dimension
            reference_points_cam = torch.bmm(
                reference_points_homogenous, cam_T_lidar_tensor[cam_idx])[..., 0:3]

            # normalize -> direction vector instead of point in 3d space
            reference_points_cam = reference_points_cam / \
                torch.norm(reference_points_cam, dim=-1, keepdim=True)

            query_position_feature = self.query_loc2latent(
                reference_points_cam)
            # bs x queries x dims -> queries x bs x dims
            query_position_feature = query_position_feature.permute(
                1, 0, 2)

            # query = query + query encoding + position feature (relative to cam)
            query_per_cam[cam_idx] = query + \
                query_pos + query_position_feature

        # the keys and queries are camera specific now, values are in global coordinates
        weighted_values, _ = self.attn(
            query=query_per_cam,
            key=feats_with_dir,
            value=value,
            key_pos=key_pos,
            query_pos=None,
            attn_mask=None,
            key_padding_mask=None,
            need_weights=False,
            attn_func=self._attn_weights_only_dot_prod_attn
        )

        # TODO refactor different versions of reference_points
        pos_feat = self.position_encoder_out(
            inverse_sigmoid(reference_points_orig)).permute(1, 0, 2)

        return identity + self.dropout_layer(self.proj_drop(weighted_values)) + pos_feat


@ATTENTION.register_module()
class NoProjectCrossAttention(QueryValueProjectCrossAttention):
    """A cross attention module without sensor-relative queries"""

    def __init__(self,
                 embed_dims,
                 num_heads,
                 pc_range=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        """@See base class"""

        # dont initialize parent (dont initialize unwanted layers / modules)
        BaseModule.__init__(self, init_cfg)

        if 'dropout' in kwargs:
            warnings.warn('The arguments `dropout` in MultiheadAttention '
                          'has been deprecated, now you can separately '
                          'set `attn_drop`(float), proj_drop(float), '
                          'and `dropout_layer`(dict) ')
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.pc_range = pc_range

        self.position_encoder_out = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.dir2latent = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.attn = CustomMultiheadAttention(embed_dims, num_heads, attn_drop,
                                             **kwargs)
        if self.batch_first:
            raise NotImplementedError("batch first mode is not yet supported")
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    @ deprecated_api_warning({'residual': 'identity'},
                             cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                reference_points=None,
                img_metas=None,
                reg_branches=None,
                **kwargs):
        """@see base class for details."""

        if identity is None:
            identity = query

        # input shape: bs x cam x dims x h x w
        # permute -> cams x h x w x bs  x dims
        # -> cams x h * w x bs x dim
        key_pos = key_pos.permute(1, 3, 4, 0, 2)
        key_pos = key_pos.reshape(
            key_pos.shape[0], -1, key_pos.shape[3], key_pos.shape[4])

        value = value.permute(1, 3, 4, 0, 2)
        value = value.reshape(
            value.shape[0], -1, value.shape[3], value.shape[4])

        # feats (vals and keys) as well as encoding come in shape
        # cams x patches x bs x dims
        CAMS, PATCHES_PER_CAM, BS = value.shape[0:3]
        QUERIES = reference_points.shape[1]

        # reference points from sigmoid space to cartesian:
        reference_points_orig = reference_points.clone()
        # Due to the inplace modificiations a clone is needed (autograd anomaly)
        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

        query_per_cam = torch.empty(
            (CAMS, QUERIES, BS, self.embed_dims), device=value.device)

        # create latent pos encoding for keys
        key_pos_feature = self.dir2latent(key_pos)

        feats_with_dir = value + key_pos_feature

        # all queries are the same (since we dont project)
        for cam_idx in range(CAMS):
            query_per_cam[cam_idx] = query + query_pos

        weighted_values, _ = self.attn(
            query=query_per_cam,
            key=feats_with_dir,
            value=value,
            key_pos=key_pos,
            query_pos=None,
            attn_mask=None,
            key_padding_mask=None,
            need_weights=False,
            attn_func=self._attn_weights_only_dot_prod_attn
        )

        # TODO refactor different versions of reference_points
        pos_feat = self.position_encoder_out(
            inverse_sigmoid(reference_points_orig)).permute(1, 0, 2)

        return identity + self.dropout_layer(self.proj_drop(weighted_values)) + pos_feat


@ATTENTION.register_module()
class QueryCenterOnlyProjectCrossAttention(QueryCenterValueProjectCrossAttention):
    """A cross attention module that forces query centers to be in front of the camera plane and uses no value projection"""

    def __init__(self,
                 embed_dims,
                 num_heads,
                 pc_range=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        """@See base class for details"""

        # TODO refactor
        # dont initialize parent (dont initialize unwanted layers / modules)
        BaseModule.__init__(self, init_cfg)


        if 'dropout' in kwargs:
            warnings.warn('The arguments `dropout` in MultiheadAttention '
                          'has been deprecated, now you can separately '
                          'set `attn_drop`(float), proj_drop(float), '
                          'and `dropout_layer`(dict) ')
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.pc_range = pc_range

        self.query_loc2latent = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.position_encoder_out = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.attn = CustomMultiheadAttention(embed_dims, num_heads, attn_drop,
                                             **kwargs)
        if self.batch_first:
            raise NotImplementedError("batch first mode is not yet supported")

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                reference_points=None,
                img_metas=None,
                reg_branches=None,
                **kwargs):
        """@See base for details"""

        if identity is None:
            identity = query

        if len(key_pos.shape) == 4:
            # multi scale (is h*w already)
            key_pos = key_pos.permute(1, 3, 0, 2)
            value = value.permute(1, 3, 0, 2)
            key = key.permute(1, 3, 0, 2)

        else:
            # input shape: bs x cam x dims x h x w
            # permute -> cams x h x w x bs  x dims
            # -> cams x h * w x bs x dim
            key_pos = key_pos.permute(1, 3, 4, 0, 2)
            key_pos = key_pos.reshape(
                key_pos.shape[0], -1, key_pos.shape[3], key_pos.shape[4])

            value = value.permute(1, 3, 4, 0, 2)
            value = value.reshape(
                value.shape[0], -1, value.shape[3], value.shape[4])

            key = key.permute(1, 3, 4, 0, 2)
            key = key.reshape(key.shape[0], -1, key.shape[3], key.shape[4])

        # feats (vals and keys) as well as encoding come in shape
        # cams x patches x bs x dims
        CAMS, PATCHES_PER_CAM, BS = value.shape[0:3]
        QUERIES = reference_points.shape[1]

        # TODO refactor to avoid the loop
        # bs x cams x 4 x 4 contains transforms for each batch and each cam
        cam_T_lidar_tensor = torch.empty(
            (BS, CAMS, 4, 4), device=value.device, requires_grad=False)

        for i, meta in enumerate(img_metas):  # over batch dim
            cam_T_lidar_cams = torch.tensor(np.stack(
                meta['lidar2cam']), dtype=torch.float, device=value.device, requires_grad=False)
            cam_T_lidar_tensor[i] = cam_T_lidar_cams

        # for multiplication use (A*B) == (B^T * A^T)
        # bs x cams x 4_1, 4_2 -> cams x bs x 4_2 x 4_1 (transpose / permute)
        cam_T_lidar_tensor = cam_T_lidar_tensor.permute(1, 0, 3, 2)

        # reference points from sigmoid space to cartesian:
        reference_points_orig = reference_points.clone()
        # Due to the inplace modificiations a clone is needed (autograd anomaly)
        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

        # The ones are just appended but do not need a gradient flow
        ones_query = torch.ones(
            (BS, QUERIES, 1), device=reference_points.device, requires_grad=False)

        reference_points_homogenous = torch.cat(
            [reference_points, ones_query], dim=-1)

        query_per_cam = torch.empty(
            (CAMS, QUERIES, BS, self.embed_dims), device=value.device)

        # create latent pos encoding for keys
        key_pos_feature = self.query_loc2latent(key_pos)
        feats_with_dir = value + key_pos_feature

        # for each cam project query ref points
        cam_front_facing = torch.tensor(
            [0, 0, 1], dtype=torch.float, device=value.device, requires_grad=False)

        cam_front_facing = cam_front_facing.repeat(BS, 1)
        # add dim at end
        cam_front_facing = cam_front_facing.unsqueeze(-1)

        query_masks = torch.empty(
            (CAMS, QUERIES, BS, 1), device=query.device, requires_grad=False, dtype=torch.bool)

        for cam_idx in range(CAMS):
            # projections might be different for each element in a batch
            # apply to entire batch
            # remove homogenous dimension
            reference_points_cam = torch.bmm(
                reference_points_homogenous, cam_T_lidar_tensor[cam_idx])[..., 0:3]

            # normalize -> direction vector instead of point in 3d space
            reference_points_cam = reference_points_cam / \
                torch.norm(reference_points_cam, dim=-1, keepdim=True)

            scalar_prod_query_cam = torch.bmm(
                reference_points_cam, cam_front_facing)
            # bs x q x 1 -> q x bs x 1
            scalar_prod_query_cam = scalar_prod_query_cam.permute(1, 0, 2)
            query_masks[cam_idx] = scalar_prod_query_cam >= 0.0

            query_position_feature = self.query_loc2latent(
                reference_points_cam)
            # bs x queries x dims -> queries x bs x dims
            query_position_feature = query_position_feature.permute(
                1, 0, 2)

            # query = query + query encoding + position feature (relative to cam)
            query_per_cam[cam_idx] = query + \
                query_pos + query_position_feature

        # the keys and queries are camera specific now, values are in global coordinates
        weighted_values, _ = self.attn(
            query=query_per_cam,
            key=feats_with_dir,
            value=value,
            key_pos=key_pos,
            query_pos=None,
            query_masks=query_masks,
            attn_mask=None,
            key_padding_mask=None,
            need_weights=False,
            attn_func=self._attn_weights_query_center_only_dot_prod_attn
        )

        # TODO refactor different versions of reference_points
        pos_feat = self.position_encoder_out(
            inverse_sigmoid(reference_points_orig)).permute(1, 0, 2)

        return identity + self.dropout_layer(self.proj_drop(weighted_values)) + pos_feat


@ATTENTION.register_module()
class QueryValueProjectCrossAttentionGlobalSoftmax(QueryValueProjectCrossAttention):
    """A cross attention module that uses a global instead of a per sensor softmax for attention"""

    def _attn_weights_only_dot_prod_attn(self,
                                         q: Tensor,
                                         k: Tensor,
                                         v: Tensor,
                                         attn_mask: Optional[Tensor] = None,
                                         dropout_p: float = 0.0,
                                         **kwargs) -> Tuple[Tensor, Tensor]:

        CAMS, B, Q, E = q.shape
        _, _, Ns, E = v.shape

        q = q / math.sqrt(E)
        k = k / math.sqrt(E)

        patches_per_img = k.shape[2]

        attn_full = torch.empty(
            (CAMS, B, Q, patches_per_img), device=q.device)

        for cam_idx in range(CAMS):
            # perform the attention for each query in each camera
            # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
            attn = torch.bmm(q[cam_idx], k[cam_idx].transpose(-2, -1))

            if dropout_p > 0.0:
                attn = F.dropout(attn, p=dropout_p)

            attn_full[cam_idx] = attn

        # cams x b x patches x dims -> b x cams x patches x dims
        v = v.permute(1, 0, 2, 3)
        v = v.reshape(B, -1, E)

        # cams x B x Nt x patches -> B x Nt x cams x patches
        attn_full = attn_full.permute(1, 2, 0, 3)
        attn_full = attn_full.reshape(B, Q, -1)
        attn_full = attn_full.softmax(dim=-1)

        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(attn_full, v)
        return output, attn_full
