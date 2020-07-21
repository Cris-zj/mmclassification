_base_ = [
    '../_base_/models/osnet_1.0_reid_fc512.py',
    '../_base_/datasets/market1501_bs64.py',
    '../_base_/schedules/market1501_bs64_250e_coslr.py',
]
# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'pretrained/osnet_x1_0_imagenet.pth'
resume_from = 'work_dirs/osnet_1.0_batch64/epoch_10.pth'
workflow = [('train', 1)]
