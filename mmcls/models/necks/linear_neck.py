import torch
import torch.nn as nn

from mmcv.cnn import normal_init, constant_init
from ..builder import NECKS


@NECKS.register_module()
class LinearNeck(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels):
        super(LinearNeck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._init_layers()
    
    def _init_layers(self):
        self.fc = nn.Linear(self.in_channels, self.out_channels)
        self.bn = nn.BatchNorm1d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def init_weights(self):
        normal_init(self.fc, mean=0, std=0.01, bias=0)
        constant_init(self.bn, 1)
    
    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.relu(self.bn(self.fc(x))) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.relu(self.bn(self.fc(inputs)))
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs


