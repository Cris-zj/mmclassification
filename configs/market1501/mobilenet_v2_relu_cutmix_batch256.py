_base_ = [
    '../_base_/models/mobilenet_v2_1x_relu_reid_fc128.py',
    '../_base_/schedules/market1501_bs256_250e_coslr.py',
]

# dataset settings
dataset_type = 'Market1501'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, 128)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='CutMix', beta=1.0, cutmix_prob=0.5),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, 128)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label', 'camid'])
]
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_prefix='data/market1501/bounding_box_train/',
        pipeline=train_pipeline),
    query=dict(
        type=dataset_type,
        data_prefix='data/market1501/query',
        test_mode=True,
        pipeline=test_pipeline),
    gallery=dict(
        type=dataset_type,
        data_prefix='data/market1501/bounding_box_test',
        test_mode=True,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='euclidean', ranks=[1, 5, 10, 20])

# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
