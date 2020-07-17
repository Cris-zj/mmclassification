_base_ = [
    '../_base_/models/osnet_1.0_reid_fc512.py',
    '../_base_/datasets/market1501_bs64.py',
    '../_base_/schedules/market1501_bs64_250e_coslr.py',
    '../_base_/default_runtime.py'
]
