# optimizer
optimizer = dict(type='Adam', lr=0.0015, weight_decay=5e-4,
                 betas=(0.9, 0.999), amsgrad=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnealing', min_lr=0)
total_epochs = 250
