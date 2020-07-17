_base_ = [
    '../_base_/models/resnet50_reid_fc512.py',
    '../_base_/datasets/market1501_bs64.py',
    '../_base_/schedules/market1501_bs64_60e.py',
    '../_base_/default_runtime.py'
]
