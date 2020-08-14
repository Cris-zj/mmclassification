import math
import random
from collections import deque

import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class RandomCrop(object):
    """Crop the given Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. If a sequence of length 4 is provided, it is used to
            pad left, top, right, bottom borders respectively.  If a sequence
            of length 2 is provided, it is used to pad left/right, top/bottom
            borders, respectively. Default: None, which means no padding.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
            Default: False.
        pad_val (Number | Sequence[Number]): Pixel pad_val value for constant
            fill. If a tuple of length 3, it is used to pad_val R, G, B
            channels respectively. Default: 0.
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Default: constant.
            -constant: Pads with a constant value, this value is specified
                with pad_val.
            -edge: pads with the last value at the edge of the image.
            -reflect: Pads with reflection of image without repeating the
                last value on the edge. For example, padding [1, 2, 3, 4]
                with 2 elements on both sides in reflect mode will result
                in [3, 2, 1, 2, 3, 4, 3, 2].
            -symmetric: Pads with reflection of image repeating the last
                value on the edge. For example, padding [1, 2, 3, 4] with
                2 elements on both sides in symmetric mode will result in
                [2, 1, 1, 2, 3, 4, 4, 3].
    """

    def __init__(self,
                 size,
                 padding=None,
                 pad_if_needed=False,
                 pad_val=0,
                 padding_mode='constant'):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        # check padding mode
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.pad_val = pad_val
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: Params (xmin, ymin, target_height, target_width) to be
                passed to ``crop`` for random crop.
        """
        height = img.shape[0]
        width = img.shape[1]
        target_height, target_width = output_size
        if width == target_width and height == target_height:
            return 0, 0, height, width

        xmin = random.randint(0, height - target_height)
        ymin = random.randint(0, width - target_width)
        return xmin, ymin, target_height, target_width

    def __call__(self, results):
        """
        Args:
            img (ndarray): Image to be cropped.
        """
        for key in results.get('img_fields', ['img']):
            img = results[key]
            if self.padding is not None:
                img = mmcv.impad(
                    img, padding=self.padding, pad_val=self.pad_val)

            # pad the height if needed
            if self.pad_if_needed and img.shape[0] < self.size[0]:
                img = mmcv.impad(
                    img,
                    padding=(0, self.size[0] - img.shape[0], 0,
                             self.size[0] - img.shape[0]),
                    pad_val=self.pad_val,
                    padding_mode=self.padding_mode)

            # pad the width if needed
            if self.pad_if_needed and img.shape[1] < self.size[1]:
                img = mmcv.impad(
                    img,
                    padding=(self.size[1] - img.shape[1], 0,
                             self.size[1] - img.shape[1], 0),
                    pad_val=self.pad_val,
                    padding_mode=self.padding_mode)

            xmin, ymin, height, width = self.get_params(img, self.size)
            results[key] = mmcv.imcrop(
                img,
                np.array([ymin, xmin, ymin + width - 1, xmin + height - 1]))
        return results

    def __repr__(self):
        return (self.__class__.__name__ +
                f'(size={self.size}, padding={self.padding})')


@PIPELINES.register_module()
class RandomResizedCrop(object):
    """Crop the given image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        scale (tuple): Range of the random size of the cropped image compared
            to the original image. Default: (0.08, 1.0).
        ratio (tuple): Range of the random aspect ratio of the cropped image
            compared to the original image. Default: (3. / 4., 4. / 3.).
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Default:
            'bilinear'.
    """

    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear'):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            raise ValueError("range should be of kind (min, max). "
                             f"But received {scale}")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (ndarray): Image to be cropped.
            scale (tuple): Range of the random size of the cropped image
                compared to the original image size.
            ratio (tuple): Range of the random aspect ratio of the cropped
                image compared to the original image area.

        Returns:
            tuple: Params (xmin, ymin, target_height, target_width) to be
                passed to ``crop`` for a random sized crop.
        """
        height = img.shape[0]
        width = img.shape[1]
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            target_width = int(round(math.sqrt(target_area * aspect_ratio)))
            target_height = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_width <= width and 0 < target_height <= height:
                xmin = random.randint(0, height - target_height)
                ymin = random.randint(0, width - target_width)
                return xmin, ymin, target_height, target_width

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            target_width = width
            target_height = int(round(target_width / min(ratio)))
        elif in_ratio > max(ratio):
            target_height = height
            target_width = int(round(target_height * max(ratio)))
        else:  # whole image
            target_width = width
            target_height = height
        xmin = (height - target_height) // 2
        ymin = (width - target_width) // 2
        return xmin, ymin, target_height, target_width

    def __call__(self, results):
        """
        Args:
            img (ndarray): Image to be cropped and resized.

        Returns:
            ndarray: Randomly cropped and resized image.
        """
        for key in results.get('img_fields', ['img']):
            img = results[key]
            xmin, ymin, target_height, target_width = self.get_params(
                img, self.scale, self.ratio)
            img = mmcv.imcrop(
                img,
                np.array([
                    ymin, xmin, ymin + target_width - 1,
                    xmin + target_height - 1
                ]))
            results[key] = mmcv.imresize(
                img, tuple(self.size[::-1]), interpolation=self.interpolation)
        return results

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(size={self.size}'
        format_string += f', scale={tuple(round(s, 4) for s in self.scale)}'
        format_string += f', ratio={tuple(round(r, 4) for r in self.ratio)}'
        format_string += f', interpolation={self.interpolation})'
        return format_string


@PIPELINES.register_module()
class RandomGrayscale(object):
    """Randomly convert image to grayscale with a probability of gray_prob.

    Args:
        gray_prob (float): Probability that image should be converted to
            grayscale. Default: 0.1.

    Returns:
        ndarray: Grayscale version of the input image with probability
            gray_prob and unchanged with probability (1-gray_prob).
            - If input image is 1 channel: grayscale version is 1 channel.
            - If input image is 3 channel: grayscale version is 3 channel
                with r == g == b.

    """

    def __init__(self, gray_prob=0.1):
        self.gray_prob = gray_prob

    def __call__(self, results):
        """
        Args:
            img (ndarray): Image to be converted to grayscale.

        Returns:
            ndarray: Randomly grayscaled image.
        """
        for key in results.get('img_fields', ['img']):
            img = results[key]
            num_output_channels = img.shape[2]
            if random.random() < self.gray_prob:
                if num_output_channels > 1:
                    img = mmcv.rgb2gray(img)[:, :, None]
                    results[key] = np.dstack(
                        [img for _ in range(num_output_channels)])
                    return results
            results[key] = img
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(gray_prob={self.gray_prob})'


@PIPELINES.register_module()
class RandomFlip(object):
    """Flip the image randomly.

    Flip the image randomly based on flip probaility and flip direction.

    Args:
        flip_prob (float): probability of the image being flipped. Default: 0.5
        direction (str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    def __init__(self, flip_prob=0.5, direction='horizontal'):
        assert 0 <= flip_prob <= 1
        assert direction in ['horizontal', 'vertical']
        self.flip_prob = flip_prob
        self.direction = direction

    def __call__(self, results):
        """Call function to flip image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """
        flip = True if np.random.rand() < self.flip_prob else False
        results['flip'] = flip
        results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            for key in results.get('img_fields', ['img']):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_prob={self.flip_prob})'


@PIPELINES.register_module()
class Resize(object):
    """Resize images.

    Args:
        size (int | tuple): Images scales for resizing (h, w).
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos".
            More details can be found in `mmcv.image.geometric`.
    """

    def __init__(self, size, interpolation='bilinear'):
        assert isinstance(size, int) or (isinstance(size, tuple)
                                         and len(size) == 2)
        if isinstance(size, int):
            size = (size, size)
        assert size[0] > 0 and size[1] > 0
        assert interpolation in ("nearest", "bilinear", "bicubic", "area",
                                 "lanczos")

        self.height = size[0]
        self.width = size[1]
        self.size = size
        self.interpolation = interpolation

    def _resize_img(self, results):
        for key in results.get('img_fields', ['img']):
            img = mmcv.imresize(
                results[key],
                size=(self.width, self.height),
                interpolation=self.interpolation,
                return_scale=False)
            results[key] = img
            results['img_shape'] = img.shape

    def __call__(self, results):
        self._resize_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@PIPELINES.register_module()
class CenterCrop(object):
    """Center crop the image.

    Args:
        crop_size (int | tuple): Expected size after cropping, (h, w).

    Notes:
        If the image is smaller than the crop size, return the original image
    """

    def __init__(self, crop_size):
        assert isinstance(crop_size, int) or (isinstance(crop_size, tuple)
                                              and len(crop_size) == 2)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size

    def __call__(self, results):
        crop_height, crop_width = self.crop_size[0], self.crop_size[1]
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_height, img_width, _ = img.shape

            y1 = max(0, int(round((img_height - crop_height) / 2.)))
            x1 = max(0, int(round((img_width - crop_width) / 2.)))
            y2 = min(img_height, y1 + crop_height) - 1
            x2 = min(img_width, x1 + crop_width) - 1

            # crop the image
            img = mmcv.imcrop(img, bboxes=np.array([x1, y1, x2, y2]))
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class Normalize(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            results[key] = mmcv.imnormalize(results[key], self.mean, self.std,
                                            self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={list(self.mean)}, '
        repr_str += f'std={list(self.std)}, '
        repr_str += f'to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class RandomErase(object):
    """Randomly erases an image patch.

    Origin: `<https://github.com/zhunzhong07/Random-Erasing>`_

    Reference:
        Zhong et al. Random Erasing Data Augmentation.

    Args:
        erase_prob (float): probability that this operation takes place.
            Default is 0.5.
        scale (tuple): Range of the random size of the erased area compared
            to the original image. Default: (0.02, 0.4).
        ratio (tuple): Range of the random aspect ratio of the erased area
            compared to the original image. Default: (3. / 10., 10. / 3).
        mean (list): erasing value. Default: [0.4914, 0.4822, 0.4465].
    """

    def __init__(self,
                 erase_prob=0.5,
                 scale=(0.02, 0.4),
                 ratio=(3. / 10., 10. / 3),
                 mean=[0.4914, 0.4822, 0.4465]):
        self.erase_prob = erase_prob
        self.scale = scale
        self.ratio = ratio
        self.mean = mean

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``erase`` for a random erase.

        Args:
            img (ndarray): Image to be erased.
            scale (tuple): Range of the random size of the erased area
                compared to the original image size.
            ratio (tuple): Range of the random aspect ratio of the erased area
                compared to the original image area.

        Returns:
            tuple: Params (xmin, ymin, target_height, target_width) to be
                passed to ``erase`` for a random erase.
        """

        height = img.shape[0]
        width = img.shape[1]
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            target_height = int(round(math.sqrt(target_area * aspect_ratio)))
            target_width = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_height <= height and 0 < target_width <= width:
                xmin = random.randint(0, height - target_height)
                ymin = random.randint(0, width - target_width)
                return xmin, ymin, target_height, target_width

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            target_width = width
            target_height = int(round(target_width / min(ratio)))
        elif in_ratio > max(ratio):
            target_height = height
            target_width = int(round(target_height * max(ratio)))
        else:  # whole image
            target_width = width
            target_height = height
        xmin = (height - target_height) // 2
        ymin = (width - target_width) // 2
        return xmin, ymin, target_height, target_width

    def __call__(self, results):
        if random.uniform(0, 1) > self.erase_prob:
            return results

        for key in results.get('img_fields', ['img']):
            img = results[key]
            xmin, ymin, h, w = self.get_params(
                img, self.scale, self.ratio)
            if img.ndim == 3:
                img[xmin:xmin + h, ymin:ymin + w, 0] = self.mean[0]
                img[xmin:xmin + h, ymin:ymin + w, 1] = self.mean[1]
                img[xmin:xmin + h, ymin:ymin + w, 2] = self.mean[2]
            else:
                img[xmin:xmin + h, ymin:ymin + w, 0] = self.mean[0]
            results[key] = img
        return results

    def __repr__(self):
        format_string = self.__class__.__name__
        format_string += f'(erase_prob={self.erase_prob}'
        format_string += f', scale={tuple(round(s, 4) for s in self.scale)}'
        format_string += f', ratio={tuple(round(r, 4) for r in self.ratio)}'
        format_string += f', mean={tuple(round(m, 4) for m in self.mean)})'
        return format_string


@PIPELINES.register_module()
class RandomPatch(object):
    """Random patch data augmentation.

    For each input image, RandomPatch
        1) extracts a random patch and stores the patch in the patch pool;
        2) randomly selects a patch from the patch pool and pastes it on the
           input (at random position) to simulate occlusion.

    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019. # noqa: E501
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. arXiv preprint, 2019.

    Arg:

    """

    def __init__(self,
                 patch_prob=0.5,
                 scale=(0.01, 0.5),
                 ratio=(1. / 10., 10.),
                 rotate_prob=0.5,
                 rotate_angle=(-10, 10),
                 flip_prob=0.5,
                 total_num_samples=2,
                 min_num_samples=100,
                 pool_capacity=50000):
        self.patch_prob = patch_prob
        self.scale = scale
        self.ratio = ratio
        self.rotate_prob = rotate_prob
        self.rotate_angle = rotate_angle
        self.flip_prob = flip_prob
        self.total_num_samples = total_num_samples
        self.min_num_samples = min_num_samples
        self.pool_capacity = pool_capacity
        self.patch_pool = deque(maxlen=pool_capacity)

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``patch`` for a random patch.

        Args:
            img (ndarray): Image to be crop.
            scale (tuple): Range of the random size of the erased area
                compared to the original image size.
            ratio (tuple): Range of the random aspect ratio of the erased area
                compared to the original image area.

        Returns:
            tuple: Params (xmin, ymin, target_height, target_width) to be
                passed to ``patch`` for a random patch.
        """

        height = img.shape[0]
        width = img.shape[1]
        area = height * width

        for _ in range(100):
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            target_height = int(round(math.sqrt(target_area * aspect_ratio)))
            target_width = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_height <= height and 0 < target_width <= width:
                xmin = random.randint(0, height - target_height)
                ymin = random.randint(0, width - target_width)
                return xmin, ymin, target_height, target_width

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            target_width = width
            target_height = int(round(target_width / min(ratio)))
        elif in_ratio > max(ratio):
            target_height = height
            target_width = int(round(target_height * max(ratio)))
        else:  # whole image
            target_width = width
            target_height = height
        xmin = (height - target_height) // 2
        ymin = (width - target_width) // 2
        return xmin, ymin, target_height, target_width

    def __call__(self, results):
        self.patch_pool.append(results.copy())
        if len(self.patch_pool) < self.min_num_samples:
            return results
        if random.uniform(0, 1) > self.patch_prob:
            return results

        sample = np.random.choice(
            self.patch_pool, self.total_num_samples - 1)[0]

        for key in results.get('img_fields', ['img']):
            img = results[key]
            sample_img = sample[key]
            xmin, ymin, h, w = self.get_params(
                sample_img, self.scale, self.ratio)
            patch = mmcv.imcrop(
                sample_img,
                np.array([
                    ymin, xmin, ymin + w - 1,
                    xmin + h - 1
                ])
            )
            if random.uniform(0, 1) > self.flip_prob:
                patch = mmcv.imflip(patch, 'horizontal')
            if random.uniform(0, 1) > self.rotate_prob:
                patch = mmcv.imrotate(
                    patch, random.randint(*self.rotate_angle))
            x1 = random.randint(0, img.shape[0] - h)
            y1 = random.randint(0, img.shape[1] - w)
            img[x1:x1 + h, y1:y1 + w] = patch
            results[key] = img

        alpha = float(h * w) / float(img.shape[0] * img.shape[1])

        gt_onehot = results['gt_onehot']
        sample_gt_onehot = sample['gt_onehot']

        target_gt_onehot = (1. - alpha) * gt_onehot + alpha * sample_gt_onehot
        results['gt_onehot'] = target_gt_onehot

        return results

    def __repr__(self):
        format_string = self.__class__.__name__
        format_string += f'(patch_prob={self.patch_prob}'
        format_string += f', scale={tuple(round(s, 4) for s in self.scale)}'
        format_string += f', ratio={tuple(round(r, 4) for r in self.ratio)}'
        format_string += f', rotate_prob={self.rotate_prob}'
        format_string += f', rotate_angle={tuple(int(r) for r in self.rotate_angle)}'  # noqa: E501
        format_string += f', flip_prob={self.flip_prob}'
        format_string += f', total_num_samples={self.total_num_samples}'
        format_string += f', min_num_samples={self.min_num_samples}'
        format_string += f', pool_capacity={self.pool_capacity}'
        return format_string


@PIPELINES.register_module()
class MixUp(object):
    """Mix two images.
    Refer to the paper for more details: https://arxiv.org/abs/1710.09412.pdf.
    Arg:
        mixup_lambda (float): the weight of mixed image.
        mixup_prob (float): Probability that this operation takes place.
            Default is 0.5.
        total_num_samples (int): The numbers of combined images.
        pool_capacity (int): maximum images in pool for selecting images.
    """

    def __init__(self,
                 mixup_lambda=0.5,
                 mixup_prob=0.5,
                 total_num_samples=2,
                 min_num_samples=100,
                 pool_capacity=50000):
        self.mixup_lambda = mixup_lambda
        self.mixup_prob = mixup_prob
        self.total_num_samples = total_num_samples
        self.min_num_samples = min_num_samples
        self.pool_capacity = pool_capacity
        self.mixup_pool = deque(maxlen=pool_capacity)

    def __call__(self, results):
        self.mixup_pool.append(results.copy())
        if len(self.mixup_pool) < self.min_num_samples:
            return results
        if random.uniform(0, 1) > self.mixup_prob:
            return results

        mixup_sample = np.random.choice(
            self.mixup_pool, self.total_num_samples - 1)[0]

        img = results['img']
        gt_onehot = results['gt_onehot']

        mixup_img = mixup_sample['img']
        mixup_gt_onehot = mixup_sample['gt_onehot']

        target_img = self.mixup_lambda * img + \
            (1 - self.mixup_lambda) * mixup_img

        target_gt_onehot = self.mixup_lambda * gt_onehot + \
            (1 - self.mixup_lambda) * mixup_gt_onehot

        results['img'] = target_img
        results['gt_onehot'] = target_gt_onehot

        return results

    def __repr__(self):
        format_string = self.__class__.__name__
        format_string += f'(mixup_lambda={self.mixup_lambda}'
        format_string += f', mixup_prob={self.mixup_prob}'
        format_string += f', total_num_samples={self.total_num_samples}'  # noqa: E501
        format_string += f', pool_capacity={self.pool_capacity})'
        return format_string


@PIPELINES.register_module()
class CutMix(object):
    """Cut and Mix two images.
    Refer to the paper for more details: https://arxiv.org/pdf/1905.04899v2.pdf. # noqa: E501
    Args:
        beta (float): The beta distribution. Default is 1.0.
        cutmix_prob (float): Probability that this operation takes place.
            Default is 0.5.
        total_num_samples (int): The numbers of combined images.
        pool_capacity (int): Maximum images in pool for selecting images.
    """

    def __init__(self,
                 beta=1.0,
                 cutmix_prob=0.5,
                 total_num_samples=2,
                 min_num_samples=100,
                 pool_capacity=50000):
        self.beta = beta
        self.cutmix_prob = cutmix_prob
        self.total_num_samples = total_num_samples
        self.min_num_samples = min_num_samples
        self.pool_capacity = pool_capacity
        self.cutmix_pool = deque(maxlen=pool_capacity)

    @staticmethod
    def get_params(img, lam):
        """Get parameters for cut and mix.

        Args:
            img (ndarray): Image to be cut and mixed.
            lam (float): The combination ratio between two images.

        Returns:
            tuple: Params (left, top, right, bottom, alpha) to be used in cutmix. # noqa: E501
        """

        height = img.shape[0]
        width = img.shape[1]
        cut_ratio = np.sqrt(1. - lam)

        cut_h = int(height * cut_ratio)
        cut_w = int(width * cut_ratio)

        cut_cy = random.randint(0, height)
        cut_cx = random.randint(0, width)

        left = np.clip(cut_cx - cut_w // 2, 0, width - 1)
        top = np.clip(cut_cy - cut_h // 2, 0, height - 1)
        right = np.clip(cut_cx + cut_w // 2, 0, width - 1)
        bottom = np.clip(cut_cy + cut_h // 2, 0, height - 1)

        alpha = 1 - float(cut_h * cut_w) / float(height * width)

        return left, top, right, bottom, alpha

    def __call__(self, results):
        self.cutmix_pool.append(results.copy())
        if len(self.cutmix_pool) < self.min_num_samples:
            return results
        if random.uniform(0, 1) > self.cutmix_prob:
            return results

        cutmix_sample = np.random.choice(
            self.cutmix_pool, self.total_num_samples - 1)[0]
        lam = random.betavariate(self.beta, self.beta)

        for key in results.get('img_fields', ['img']):
            img = results[key]
            left, top, right, bottom, alpha = self.get_params(img, lam)
            cutmix_img = cutmix_sample[key]
            img[top:bottom, left:right] = cutmix_img[top:bottom, left:right]
            results[key] = img

        gt_onehot = results['gt_onehot']
        cutmix_gt_onehot = cutmix_sample['gt_onehot']
        target_gt_onehot = alpha * gt_onehot + (1. - alpha) * cutmix_gt_onehot
        results['gt_onehot'] = target_gt_onehot

        return results

    def __repr__(self):
        format_string = self.__class__.__name__
        format_string += f'(beta={self.beta}'
        format_string += f', cutmix_prob={self.cutmix_prob}'
        format_string += f', total_num_samples={self.total_num_samples}'  # noqa: E501
        format_string += f', pool_capacity={self.pool_capacity})'
        return format_string
