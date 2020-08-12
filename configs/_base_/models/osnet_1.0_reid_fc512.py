# model settings
model = dict(
    type='ImageReID',
    pretrained=None,
    backbone=dict(
        type='OSNet',
        widen_factor=1.0,
        num_stages=4,
        out_indices=(3,),
        conv_cfg=None,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU', inplace=True),
    ),
    neck=[
        dict(type='GlobalAveragePooling'),
        dict(type='LinearNeck',
             in_channels=512,
             out_channels=512)
    ],
    head=dict(
        type='LinearClsHead',
        num_classes=751,
        in_channels=512,
        loss=dict(type='LabelSmoothLoss',
                  label_smooth_val=0.1,
                  num_classes=751,
                  loss_weight=1.0),
        topk=(1, 5),
    )
)
