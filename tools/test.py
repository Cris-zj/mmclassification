import argparse
import os
import time

import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmcls.apis import multi_gpu_test, single_gpu_test
from mmcls.core import wrap_fp16_model, compute_distance_matrix
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier, build_reid
from mmcls.utils import get_root_logger


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


def test_classifier(cfg, args, distributed):
    cfg.data.test.test_mode = True
    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
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
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        nums = []
        results = {}
        for output in outputs:
            nums.append(output['num_samples'].item())
            for topk, v in output['accuracy'].items():
                if topk not in results:
                    results[topk] = []
                results[topk].append(v.item())
        assert sum(nums) == len(dataset)
        for topk, accs in results.items():
            avg_acc = np.average(accs, weights=nums)
            print(f'\n{topk} accuracy: {avg_acc:.2f}')
    if args.out and rank == 0:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(outputs, args.out)


def test_reid(cfg, args, distributed, logger):
    cfg.data.query.test_mode = True
    cfg.data.gallery.test_mode = True
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
    model = build_reid(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    _ = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        logger.info('Extracting features from query set ...')
        query_outputs = single_gpu_test(model, query_loader,
                                        return_loss=False)
        print('\n')
        logger.info('Extracting features from gallery set ...')
        gallery_outputs = single_gpu_test(model, gallery_loader,
                                          return_loss=False)
        print('\n')
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        logger.info('Extracting features from query set ...')
        query_outputs = multi_gpu_test(model, query_loader,
                                       return_loss=False,
                                       tmpdir=args.tmpdir,
                                       gpu_collect=args.gpu_collect)
        print('\n')
        logger.info('Extracting features from gallery set ...')
        gallery_outputs = multi_gpu_test(model, gallery_loader,
                                         return_loss=False,
                                         tmpdir=args.tmpdir,
                                         gpu_collect=args.gpu_collect)
        print('\n')

    rank, _ = get_dist_info()
    if rank == 0:
        query_feats, query_pids, query_camids = [], [], []
        for output in query_outputs:
            query_feats.append(output['feature'].data.cpu())
            query_pids.append(output['pid'].data.cpu())
            query_camids.append(output['camid'].data.cpu())
        query_feats = torch.cat(query_feats, dim=0)
        query_pids = torch.cat(query_pids, dim=0).numpy()
        query_camids = torch.cat(query_camids, dim=0).numpy()

        gallery_feats, gallery_pids, gallery_camids = [], [], []
        for output in gallery_outputs:
            gallery_feats.append(output['feature'].data.cpu())
            gallery_pids.append(output['pid'].data.cpu())
            gallery_camids.append(output['camid'].data.cpu())
        gallery_feats = torch.cat(gallery_feats, dim=0)
        gallery_pids = torch.cat(gallery_pids, dim=0).numpy()
        gallery_camids = torch.cat(gallery_camids, dim=0).numpy()

        eval_cfg = cfg.get('evaluation', {})
        dist_metric = eval_cfg.get('metric', 'euclidean')
        logger.info(f'Computing distance matrix with metric = {dist_metric} ...')
        distmat = compute_distance_matrix(
            query_feats, gallery_feats, metric=dist_metric)
        distmat = distmat.numpy()

        logger.info('Computing CMC and mAP ...')
        cmc, mAP = query_datatset.evaluate(
            distmat, query_pids, query_camids, gallery_pids, gallery_camids)

        logger.info('** Results **')
        logger.info('mAP: {:.1%}'.format(mAP))
        logger.info('CMC curve:')
        ranks = eval_cfg.get('ranks', [1, 5, 10, 20])
        for i in ranks:
            logger.info('Rank-{:<3}: {:.1%}'.format(i, cmc[i - 1]))


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    
    work_dir = os.path.split(args.checkpoint)[0]
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(work_dir, f'{timestamp}-test.log')
    logger = get_root_logger(log_file=log_file)

    task_type = cfg.model.get('type').lower()
    if 'classifier' in task_type:
        test_classifier(cfg, args, distributed)
    elif 'reid' in task_type:
        test_reid(cfg, args, distributed, logger)

    
if __name__ == '__main__':
    main()
