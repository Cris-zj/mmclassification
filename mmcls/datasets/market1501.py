import os
import re
import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class Market1501(BaseDataset):

    def load_annotations(self):
        classes, class_to_idx, samples = make_dataset(
            self.data_prefix, relabel=(not self.test_mode))
        if len(samples) == 0:
            raise (RuntimeError('Found 0 files in subfolders of: '
                                f'{self.data_prefix}. '))
        self.CLASSES = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        data_infos = []
        for filename, gt_label, camid in self.samples:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            info['camid'] = np.array(camid, dtype=np.int64)
            data_infos.append(info)
        return data_infos

    def evaluate(self,
                 distmat,
                 query_pids,
                 query_camids,
                 gallery_pids,
                 gallery_camids,
                 max_rank=50):
        """Evaluation with market1501 metric
        Key: for each query identity,
             its gallery images from the same camera view are discarded.
        """
        num_query, num_gallery = distmat.shape

        if num_gallery < max_rank:
            max_rank = num_query
            print('Note: number of gallery samples is quite small, '
                  'got {}'.format(num_gallery))
        indices = np.argsort(distmat, axis=1)
        matches = (gallery_pids[indices] ==
                   query_pids[:, np.newaxis]).astype(np.int32)

        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        num_vaild_query = 0  # number of valid query

        for query_idx in range(num_query):
            # get query pid and camid
            query_pid = query_pids[query_idx]
            query_camid = query_camids[query_idx]

            # remove gallery samples
            # that have the same pid and camid with query
            order = indices[query_idx]
            remove = (gallery_pids[order] == query_pid) & \
                (gallery_camids[order] == query_camid)
            keep = np.invert(remove)

            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            raw_cmc = matches[query_idx][keep]
            # when query identity does not appear in gallery
            if not np.any(raw_cmc):
                continue

            cmc = raw_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_vaild_query += 1

            # compute average precision
            num_rel = raw_cmc.sum()
            tmp_cmc = raw_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert num_vaild_query > 0, \
            'Error: all query identities do not appear in gallery'
        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / float(num_vaild_query)
        mAP = np.mean(all_AP)

        return all_cmc, mAP


def make_dataset(root, relabel=False):
    images = []
    root = os.path.expanduser(root)
    pattern = re.compile(r'([-\d]+)_c(\d)')
    classes = set()
    for img_path in os.listdir(root):
        if os.path.splitext(img_path)[1] != '.jpg':
            continue
        pid, _ = map(int, pattern.search(img_path).groups())
        if pid == -1:
            continue  # junk images are just ignored
        classes.add(pid)
    class_to_idx = {pid: i for i, pid in enumerate(classes)}

    for img_path in os.listdir(root):
        if os.path.splitext(img_path)[1] != '.jpg':
            continue
        pid, camid = map(int, pattern.search(img_path).groups())
        if pid == -1:
            continue  # junk images are just ignored
        assert 0 <= pid <= 1501  # pid == 0 means background
        assert 1 <= camid <= 6
        camid -= 1  # index starts from 0
        if relabel:
            pid = class_to_idx[pid]
        images.append((img_path, pid, camid))
    return classes, class_to_idx, images
