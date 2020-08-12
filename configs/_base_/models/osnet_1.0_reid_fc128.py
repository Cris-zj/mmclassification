# model settings
model = dict(
    type='ImageReID',
    pretrained='pretrained/osnet_x1_0_imagenet.pth',
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
             out_channels=128)
    ],
    head=dict(
        type='LinearClsHead',
        num_classes=751,
        in_channels=128,
        loss=dict(type='LabelSmoothLoss',
                  label_smooth_val=0.1,
                  num_classes=751,
                  loss_weight=1.0),
        topk=(1, 5),
    )
)
