# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='poly',
    power=4.,
    warmup='pow',
    warmup_iters=1000,
    warmup_ratio=1.0 / 4)
total_epochs = 160