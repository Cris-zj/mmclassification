_base_ = [
    '../_base_/models/resnet50_reid_fc128.py',
    '../_base_/datasets/market1501_bs32.py',
    '../_base_/schedules/market1501_bs32_60e.py',
    '../_base_/default_runtime.py'
]
