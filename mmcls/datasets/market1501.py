import os
import re
import glob

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class Market1501(BaseDataset):

    def load_annotations(self):
        classes, class_to_idx, samples = make_dataset(data_prefix)
        if len(samples) == 0:
            raise (RuntimeError('Found 0 files in subfolders of: '
                                f'{self.data_prefix}. '))
        self.CLASSES = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        data_infos = []
        for filename, gt_label, camid in self.samples:
            nfo = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            info['camid'] = np.array(camid, dtype=np.int64)
            data_infos.append(info)
        return data_infos


def make_dataset(root):
    images = []
    root = os.path.expanduser(root)
    img_paths_list = glob.glob(os.path.join(root, '*.jpg'))
    pattern = re.compile(r'([-\d]+)_c(\d)')
    classes = set()
    for img_path in img_paths_list:
        pid, _ = map(int, pattern.search(img_path).groups())
        if pid == -1:
            continue # junk images are just ignored
        classes.add(pid)
    class_to_idx = {pid: i for i, pid in enumerate(classes)}

    for img_path in img_paths:
        pid, camid = map(int, pattern.search(img_path).groups())
        if pid == -1:
            continue # junk images are just ignored
        assert 0 <= pid <= 1501 # pid == 0 means background
        assert 1 <= camid <= 6
        camid -= 1 # index starts from 0
        if not self.test_mode:
            pid = class_to_idx[pid]
        images.append((img_path, pid, camid))
    return classes, class_to_idx, images
