# optimizer
optimizer = dict(type='Adam', lr=0.0012, weight_decay=5e-4,
                 betas=(0.9, 0.999), amsgrad=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[20])
total_epochs = 60
