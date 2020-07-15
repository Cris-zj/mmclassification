# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=[
        dict(type='GlobalAveragePooling'),
        dict(type='LinearNeck',
             in_channels=2048,
             out_channels=512)
    ],
    head=dict(
        type='LinearClsHead',
        num_classes=751,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
