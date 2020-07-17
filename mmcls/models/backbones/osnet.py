import logging

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      build_activation_layer,
                      constant_init, kaiming_init)
from mmcv.runner import load_checkpoint

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


def lite_3x3_layer(inplanes,
                   planes,
                   conv_cfg=None,
                   norm_cfg=dict(type='BN'),
                   act_cfg=dict(type='ReLU')):
    layers = []
    layers.append(build_conv_layer(conv_cfg,
                                   inplanes,
                                   planes,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   dilation=1,
                                   groups=1,
                                   bias=0))
    layers.append(build_conv_layer(conv_cfg,
                                   planes,
                                   planes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   dilation=1,
                                   groups=planes,
                                   bias=0))
    layers.append(build_norm_layer(norm_cfg, planes)[1])
    layers.append(build_activation_layer(act_cfg))
    return nn.Sequential(*layers)


class ChannelGate(nn.Module):

    def __init__(self,
                 inplanes):
        super(ChannelGate, self).__init__()
        self.inplanes = inplanes
        self.gap = nn.AdaptiveAvgPool2d(1)
        reduction = 16
        self.fc1 = nn.Conv2d(self.inplanes,
                             self.inplanes // reduction,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(self.inplanes // reduction,
                             self.inplanes,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        identity = x

        x = self.gap(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.gate_activation(x)

        return identity * x


class OSBlock(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 downsample=None,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super(OSBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.downsample = downsample
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        mid_planes = planes // self.expansion
        self.conv1x1_first = ConvModule(
            in_channels=self.inplanes,
            out_channels=mid_planes,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        multi_streams = []
        for i in range(self.expansion):
            layers = []
            for _ in range(i + 1):
                layers.append(
                    lite_3x3_layer(
                        mid_planes,
                        mid_planes,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    )
                )
            multi_streams.append(nn.Sequential(*layers))
        self.multi_streams = nn.ModuleList(multi_streams)

        self.gate_fuse = ChannelGate(mid_planes)

        self.conv1x1_last = ConvModule(
            in_channels=mid_planes,
            out_channels=self.planes,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=None
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x1 = self.conv1x1_first(x)

        out = self.gate_fuse(self.multi_streams[0](x1))
        for i in range(1, self.expansion):
            out += self.gate_fuse(self.multi_streams[i](x1))

        out = self.conv1x1_last(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


@BACKBONES.register_module()
class OSNet(BaseBackbone):

    arch_settings = {
        0.25: [16, 64, 96, 128],
        0.5: [32, 128, 192, 256],
        0.75: [48, 192, 288, 384],
        1.0: [64, 256, 384, 512]
    }

    def __init__(self,
                 widen_factor=1.0,
                 num_stages=4,
                 out_indices=(3,),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 norm_eval=False):
        super(OSNet, self).__init__()
        self.widen_factor = widen_factor
        for index in out_indices:
            if index not in range(0, 8):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 8). But received {index}')

        if frozen_stages not in range(-1, 8):
            raise ValueError('frozen_stages must be in range(-1, 8). '
                             f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_cfg

        self.inplanes = round(64 * self.widen_factor)
        self._make_stem_layer()
        self.block = OSBlock
        self.stage_blocks = self.arch_settings[widen_factor]
        self.stage_repeats = [2, 2, 2]
        self.os_layers = []
        for i, num_blocks in enumerate(self.stage_repeats):
            outplanes = self.stage_blocks[i + 1]
            reduce_spatial_size = True if i < 2 else False
            os_layer = self.make_os_layer(
                self.block,
                self.inplanes,
                outplanes,
                num_blocks,
                reduce_spatial_size)
            self.inplanes = outplanes
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, os_layer)
            self.os_layers.append(layer_name)
        layer = ConvModule(
            in_channels=self.inplanes,
            out_channels=self.inplanes,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        layer_name = f'layer{len(self.stage_repeats) + 1}'
        self.add_module(layer_name, layer)
        self.os_layers.append(layer_name)

        self._freeze_stages()

    def _make_stem_layer(self):
        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def make_os_layer(self,
                      block,
                      in_channels,
                      out_channels,
                      num_blocks,
                      reduce_spatial_size):
        downsample = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=None
        )
        layers = []
        layers.append(
            block(
                inplanes=in_channels,
                planes=out_channels,
                downsample=downsample,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
        )
        for i in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=out_channels,
                    planes=out_channels,
                    downsample=None,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    ),
                    nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
                )
            )
        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        outs = []
        for i, layer_name in enumerate(self.os_layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(OSNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
