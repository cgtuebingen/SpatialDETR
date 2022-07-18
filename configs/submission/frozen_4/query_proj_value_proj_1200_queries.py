_base_ = ['./base.py', '../../shedules/detr3d.py']

model = dict(
    pts_bbox_head=dict(
        num_query=1200,
        transformer=dict(
            pos_encoding=dict(
                _delete_=True,
                type="FixedGeometricEncoding",
                # encodings are in  sensor coordinates of each cam
                apply_global_rot=False,
            ),
            decoder=dict(
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type="QueryValueProjectCrossAttention",
                            embed_dims=256,
                            num_heads=8,
                            pc_range={{_base_.point_cloud_range}},
                            dropout=0.1,
                        ),
                    ],
                )
            )
        )
    )
)
