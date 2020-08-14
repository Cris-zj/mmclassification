# model settings
model = dict(
    type='ImageReID',
    pretrained='pretrained/mobilenetv2_relu.pth',
    backbone=dict(type='MobileNetV2',
                  widen_factor=1.0,
                  act_cfg=dict(type='ReLU')),
    neck=[
        dict(type='GlobalAveragePooling'),
        dict(type='LinearNeck',
             in_channels=1280,
             out_channels=128)
    ],
    head=dict(
        type='LinearSoftClsHead',
        num_classes=751,
        in_channels=128,
        loss=dict(type='SoftLabelSmoothLoss',
                  label_smooth_val=0.1,
                  num_classes=751,
                  loss_weight=1.0),
        topk=(1, 5),
    )
)
