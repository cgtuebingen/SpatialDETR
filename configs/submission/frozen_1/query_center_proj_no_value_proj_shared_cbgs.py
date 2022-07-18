_base_ = ['./base.py', '../../shedules/detr3d.py']

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            pos_encoding=dict(
                _delete_=True,
                type="FixedGeometricEncoding",
                # encodings are in  sensor coordinates of each cam
                apply_global_rot=False,
            ),
            decoder=dict(
                shared=True,
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type="QueryCenterOnlyProjectCrossAttention",
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

_data_root = "data/nuscenes/"
_img_norm_cfg = dict(mean=[103.530, 116.280, 123.675],
                     std=[1.0, 1.0, 1.0], to_rgb=False)


# For nuScenes we usually do 10-class detection
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]


train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
    ),
    dict(type="ObjectRangeFilter", point_cloud_range={{_base_.point_cloud_range}}),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="NormalizeMultiviewImage", **_img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type="Collect3D",
        keys=["gt_bboxes_3d", "gt_labels_3d", "img"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "lidar2img",
            "depth2img",
            "cam2img",
            "pad_shape",
            "scale_factor",
            "flip",
            "pcd_horizontal_flip",
            "pcd_vertical_flip",
            "box_mode_3d",
            "box_type_3d",
            "img_norm_cfg",
            "pcd_trans",
            "sample_idx",
            "pcd_scale_factor",
            "pcd_rotation",
            "pts_filename",
            "transformation_3d_flow",
            "cam_intrinsic",
            "lidar2cam",
        ),
    ),
]


input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False
)

data = dict(
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type="CustomNuScenesDataset",
            data_root=_data_root,
            ann_file=_data_root + 'nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            use_valid_flag=True,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR'),
    ),
)
