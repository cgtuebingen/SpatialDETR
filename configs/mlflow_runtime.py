_base_ = '../mmdetection3d/configs/_base_/default_runtime.py'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='MlflowLoggerHook')
    ]
)
