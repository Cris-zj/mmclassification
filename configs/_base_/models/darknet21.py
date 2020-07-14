# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='DarkNet',
        depth=21,
        num_stages=5,
        out_indices=(4, ),
        conv_cfg=None,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='LeakyReLU',
                     negative_slope=0.1, inplace=True)),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))