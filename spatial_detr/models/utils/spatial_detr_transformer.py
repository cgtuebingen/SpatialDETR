import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER
from torch.nn import ModuleList

from spatial_detr.thirdparty.detr3d.util import inverse_sigmoid

@TRANSFORMER.register_module()
class SpatialDETRTransformer(BaseModule):
    """Transformer for SpatialDETR
    """

    def __init__(
        self,
        pos_encoding=None,
        decoder=None,
        **kwargs
    ):
        """Creates the SpatialDETRTransformer.

        Parameters
        ----------
        pos_encoding : dict, optional
            config for the positional encoding, by default None
        decoder : dict, optional
            configuration of the decoder, by default None
        """
        super(SpatialDETRTransformer, self).__init__(**kwargs)
        self.pos_encoding_cfg = pos_encoding
        self.decoder_cfg = decoder
        self.embed_dims = None
        self.init_layers()

    def init_layers(self):
        """Build all layers."""

        self.pos_encoding = build_positional_encoding(
            self.pos_encoding_cfg)

        self.decoder = build_transformer_layer_sequence(self.decoder_cfg)
        self.embed_dims = self.decoder.embed_dims
        # TODO allow to pass via config for more complex encoder
        self.latent2ref = nn.Linear(self.embed_dims, 3)

    def init_weights(self):
        """Initialize the transformer weights."""
        # as usually use xavier uniform for all transformer params
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        xavier_init(self.latent2ref, distribution="uniform", bias=0.0)

    def forward(
        self,
        mlvl_img_feats,
        query_embed,
        reg_branches=None,
        **kwargs
    ):
        """Forward function of the transformer. Is compatible to the detr3d interface.

        Parameters
        ----------
        mlvl_img_feats : list 
            Features from camera backbone
        query_embed : tensor
            latent query embedding (query + query_embed) concatenated
        reg_branches : nn.module, optional
            Regression branches latent -> box, by default None

        Returns
        -------
        torch.tensor
        Object queries fitted to data
        """

        
        # query embed contains query + query pos embed weights
        assert query_embed is not None

        # we keep the DETR3D interface here for simplicity
        # full credits belong to the authors
        # BS is the same for all feats
        BS = mlvl_img_feats[0].size(0)

        # query embed contains query + query pos embed weights
        query_pos_encoding, query_prior = torch.split(query_embed, self.embed_dims, dim=1)
        
        # encodings are the same for each elem in the batch
        query_pos_encoding = query_pos_encoding.unsqueeze(0)
        query_pos_encoding = query_pos_encoding.expand(BS, -1, -1)

        # same holds for the query priors
        query_prior = query_prior.unsqueeze(0).expand(BS, -1, -1)
        ref_points = self.latent2ref(query_pos_encoding)

        # convert to sigmoid space
        ref_points = ref_points.sigmoid()
        ref_points_prior = ref_points

        query_prior = query_prior.permute(1, 0, 2)
        query_pos_encoding = query_pos_encoding.permute(1, 0, 2)

        # Positional encoding
        input_img_h, input_img_w = kwargs["img_metas"][0]["ori_shape"][0:2]
        padded_h, padded_w, _ = kwargs["img_metas"][0]["pad_shape"][0]
        # refactor: we assume all imgs to have the same shape
        # (h,w,channels,cams)
        img_position_mask = torch.ones(
            (BS, padded_h, padded_w), device=mlvl_img_feats[0].device, requires_grad=False)
        # facebook convention: 0 is valid, 1 is invalid
        img_position_mask[:, :input_img_h, :input_img_w] = 0

        pos_encodings = []
        # build pos encoding for each feature lvl:
        for lvl in range(len(mlvl_img_feats)):

            feature_height = mlvl_img_feats[lvl].shape[-2]
            feature_width = mlvl_img_feats[lvl].shape[-1]

            # interpolate masks to have the same spatial shape with feats per cam
            # squeeze is needed since interpolate expects a channel dimension
            img_position_mask_feature = (
                F.interpolate(img_position_mask.unsqueeze(
                    1), size=(feature_height, feature_width))
                .to(torch.bool)
                .squeeze(1)
            )

            # cams x bs x dim x h x w
            pos_encoding = self.pos_encoding(
                img_position_mask_feature, kwargs["img_metas"])

            # permute:
            # cams x bs x dim x h x w -> bs x c x d x h x w
            pos_encoding = pos_encoding.permute(1, 0, 2, 3, 4)

            pos_encodings.append(pos_encoding)

        # run decoder
        inter_queries, inter_ref_points = self.decoder(
            query=query_prior,
            key=mlvl_img_feats,  # will be set to feats
            value=mlvl_img_feats,
            query_pos=query_pos_encoding,
            key_pos=pos_encodings,
            ref_points=ref_points,
            reg_branches=reg_branches,
            **kwargs
        )

        return inter_queries, ref_points_prior, inter_ref_points


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MixedScaleSpatialDETRTransformerDecoder(BaseModule):
    """
    Decoder that uses different scale features depending on the decoder layer
    """

    def __init__(self, shared=False, switch_scale_layer_num=5, return_intermediate=False, transformerlayers=None, num_layers=None, init_cfg=None):
        """
        Creates the MixedScaleDetr3DTransformerDecoder

        Parameters
        ----------
        shared : bool, optional
            Whether to share decoder layers, by default False
        switch_scale_layer_num : int, optional
            Layer at which swich should be performed, by default 5
            -1 to use last scale for all layers
        return_intermediate : bool, optional
            Whether to return queries for all layers, by default False
        transformerlayers : dict, optional
            Configuration of decoder layers, by default None
        num_layers : int, optional
            Amount of decoder layers, by default None
        init_cfg : dict, optional
            Weight initialization config, by default None
        """
        super(MixedScaleSpatialDETRTransformerDecoder,
              self).__init__(init_cfg)

        if isinstance(transformerlayers, dict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)
            ]
        else:
            assert isinstance(transformerlayers, list) and \
                len(transformerlayers) == num_layers
        self.num_layers = num_layers
        self.layers = ModuleList()

        if shared:
            # the first layer shall not be shared (according to perceiver)
            layer = build_transformer_layer(transformerlayers[1])
            for i in range(num_layers):
                if i == 0:
                    self.layers.append(
                        build_transformer_layer(transformerlayers[i]))
                else:
                    self.layers.append(layer)
        else:
            for i in range(num_layers):
                self.layers.append(
                    build_transformer_layer(transformerlayers[i]))

        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm

        self.return_intermediate_queries = return_intermediate
        self.switch_scale_layer_num = switch_scale_layer_num

    def forward(self,
                *args,
                query,
                value,
                key,
                key_pos,
                ref_points=None,
                reg_branches=None,
                **kwargs):
        """Forward function of decoder

        Parameters
        ----------
        query : torch.tensor
            Object queries
        value :  torch.tensor
            Camera features
        key :  torch.tensor
            Camera features (typically the same as value)
        key_pos :  torch.tensor
            Positional encoding for the camera features
        ref_points :  torch.tensor, optional
            Initial query locations, by default None
        reg_branches : nn.Module, optional
            mlp latent query -> box

        Returns
        -------
        Updated object queries using the camera data

        """

        intermediate_queries = []
        intermediate_ref_points = []
        for lid, layer in enumerate(self.layers):

            if lid >= self.switch_scale_layer_num:
                # use higher scale for last layers
                scale_idx = 0
            else:

                scale_idx = 1

            ref_points_input = ref_points
            query = layer(
                *args,
                query,
                value=value[scale_idx],
                key=key[scale_idx],
                key_pos=key_pos[scale_idx],
                reference_points=ref_points_input,
                **kwargs)
            query = query.permute(1, 0, 2)

            if reg_branches is not None:
                # infert box features from latent query
                obb = reg_branches[lid](query)

                # ref points are 3d center of obj -> 3dim
                assert ref_points.shape[-1] == 3

                updated_ref_points = torch.zeros_like(ref_points)

                # update the ref points with old + layer delta
                updated_ref_points[..., :3] = obb[
                    ..., :3] + inverse_sigmoid(ref_points[..., :3])
                updated_ref_points = updated_ref_points.sigmoid()

                ref_points = updated_ref_points.detach()

            query = query.permute(1, 0, 2)
            if self.return_intermediate_queries:
                # store intermediate outs for loss at each layer
                intermediate_queries.append(query)
                intermediate_ref_points.append(ref_points)

        if self.return_intermediate_queries:
            return torch.stack(intermediate_queries), torch.stack(
                intermediate_ref_points)

        return query, ref_points