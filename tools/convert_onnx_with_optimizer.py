"""
ONNX exporting is currently support for one-stage detector.

"""
import argparse
import torch
import onnx
from onnx import optimizer

import mmcv
from mmcv.runner import load_checkpoint
from mmcls.models import build_classifier, build_reid


def parse_args():
    """add some parameters."""
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path',
                        default=None)
    parser.add_argument('--out', help='output ONNX file', default=None)
    parser.add_argument('--checkpoint', help='checkpoint file of the model',
                        default=None)
    parser.add_argument('--shape', type=int, nargs='+', default=[480],
                        help='input image size')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if len(args.shape) == 1:
        img_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        img_shape = (1, 3) + tuple(args.shape)
    elif len(args.shape) == 4:
        img_shape = tuple(args.shape)
    else:
        raise ValueError('invalid input shape')
    dummy_input = torch.randn(*img_shape, device='cuda')
    cfg = mmcv.Config.fromfile(args.config)
    model = build_reid(cfg.model).cuda()
    model.eval()
    if args.checkpoint:
        print('load checkpoint')
        _ = load_checkpoint(model, args.checkpoint)
    model.forward = model.extract_feat
    torch.onnx.export(model, dummy_input, args.out, verbose=True, keep_initializers_as_inputs=True)
    print('Convert to onnx model successfully!')

    print("Start optimize ONNX model for inference:")
    passes = ['eliminate_identity',
              'fuse_consecutive_squeezes',
              'fuse_consecutive_transposes',
              'eliminate_nop_pad',
              'eliminate_nop_transpose',
              'eliminate_unused_initializer',
              'extract_constant_to_initializer',
              'fuse_bn_into_conv',
              'fuse_transpose_into_gemm']
    for i in range(len(passes)):
        print("%s.%s" % (i, passes[i]))
    original_model_path = args.out
    optimized_model_path = args.out[:-5] + '_sim' + args.out[-5:]
    original_model = onnx.load(original_model_path)
    optimized_model = optimizer.optimize(original_model, passes)
    onnx.save_model(optimized_model, optimized_model_path)
    print("Optimize Finished!")
    print("Please check new model in:", optimized_model_path)

if __name__ == '__main__':
    main()
