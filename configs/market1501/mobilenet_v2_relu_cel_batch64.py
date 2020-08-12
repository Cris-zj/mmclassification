_base_ = [
    '../_base_/models/mobilenet_v2_1x_relu_reid_fc128_cel.py',
    '../_base_/datasets/market1501_bs64.py',
    '../_base_/schedules/market1501_bs64_250e_coslr.py',
    '../_base_/default_runtime.py'
]
