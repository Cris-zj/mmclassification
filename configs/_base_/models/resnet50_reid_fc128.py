# model settings
model = dict(
    type='ImageReID',
    pretrained='pretrained/resnet50-19c8e357.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        strides=(1, 2, 2, 1),
        out_indices=(3, ),
        style='pytorch'),
    neck=[
        dict(type='GlobalAveragePooling'),
        dict(type='LinearNeck',
             in_channels=2048,
             out_channels=128)
    ],
    head=dict(
        type='LinearClsHead',
        num_classes=751,
        in_channels=128,
        loss=dict(type='LabelSmoothLoss',
                  label_smooth_val=0.1,
                  num_classes=751,
                  loss_weight=1.0
            ),
        topk=(1, 5),
    )
)
