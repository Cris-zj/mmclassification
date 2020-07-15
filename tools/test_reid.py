import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmcls.apis import multi_gpu_test, single_gpu_test
from mmcls.core import wrap_fp16_model, compute_distance_matrix
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier


def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.query.test_mode = True
    cfg.data.gallery.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    query_datatset = build_dataset(cfg.data.query)
    query_loader = build_dataloader(
        query_datatset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=False)

    gallery_dataset = build_dataset(cfg.data.gallery)
    gallery_loader = build_dataloader(
        gallery_dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=False)

    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    _ = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        query_outputs = single_gpu_test(model, query_loader)
        gallery_outputs = single_gpu_test(model, gallery_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        query_outputs = multi_gpu_test(model, query_loader, args.tmpdir,
                                       args.gpu_collect)
        gallery_outputs = multi_gpu_test(model, gallery_loader, args.tmpdir,
                                         args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        query_feats, query_pids, query_camids = [], [], []
        for output in query_outputs:
            query_feats.append(output['feature'].items())
            query_pids.append(output['pid'].items())
            query_camids.append(output['camid'].items())

        gallery_feats, gallery_pids, gallery_camids = [], [], []
        for output in gallery_outputs:
            gallery_feats.append(output['feature'].items())
            gallery_pids.append(output['pid'].items())
            gallery_camids.append(output['camid'].items())

        eval_cfg = cfg.get('evaluation', {})
        dist_metric = eval_cfg.get('metric', 'euclidean')
        print(f'Computing distance matrix with metric = {dist_metric} ...')
        distmat = compute_distance_matrix(
            query_feats, gallery_feats, metric=dist_metric)

        print('Computing CMC and mAP ...')
        cmc, mAP = query_datatset.evaluate(
            distmat, query_pids, query_camids, gallery_pids, gallery_camids)

        print('** Results **')
        print('mAP: {:.1f}'.format(mAP))
        print('CMC curve')
        ranks = eval_cfg.get('ranks', [1, 5, 10, 20])
        for i in ranks:
            print('Rank-{:<3}: {:.1%}'.format(i, cmc[i - 1]))


if __name__ == '__main__':
    main()
