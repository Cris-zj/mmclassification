import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .cls_head import ClsHead


@HEADS.register_module()
class LinearSoftClsHead(ClsHead):
    """Linear classifier head with multi-label.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss=dict(type='SoftCrossEntropyLoss', loss_weight=1.0),
                 topk=(1, )):
        super(LinearSoftClsHead, self).__init__(loss=loss, topk=topk)
        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self._init_layers()

    def _init_layers(self):
        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        normal_init(self.fc, mean=0, std=0.01, bias=0)

    def loss(self, cls_score, gt_label, gt_onehot):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        loss = self.compute_loss(cls_score, gt_onehot, avg_factor=num_samples)

        # compute accuracy
        acc = self.compute_accuracy(cls_score, gt_label)
        assert len(acc) == len(self.topk)
        losses['loss'] = loss
        losses['accuracy'] = {f'top-{k}': a for k, a in zip(self.topk, acc)}
        losses['num_samples'] = loss.new(1).fill_(num_samples)
        return losses

    def forward_train(self, x, gt_label, gt_onehot):
        cls_score = self.fc(x)
        losses = self.loss(cls_score, gt_label, gt_onehot)
        return losses
