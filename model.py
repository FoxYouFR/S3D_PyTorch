import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SepInc(nn.Module):
    def __init__(
            self,
            input_dim,
            num_outputs_0_0a,
            num_outputs_1_0a,
            num_outputs_1_0b,
            num_outputs_2_0a,
            num_outputs_2_0b,
            num_outputs_3_0b,
            gating_fn=None
            ):
        self.gating_fn = gating_fn
        self.use_gating = gating_fn is not None
        super(SepInc, self).__init__()
        self.branch_0 = nn.Conv3d(input_dim, num_outputs_0_0a, [1, 1, 1])
        self.branch_1_a = nn.Conv3d(input_dim, num_outputs_1_0a, [1, 1, 1])
        self.branch_1_b = SepConv(num_outputs_1_0a, num_outputs_1_0b, [3, 3, 3], padding=1)
        self.branch_2_a = nn.Conv3d(input_dim, num_outputs_2_0a, [1, 1, 1])
        self.branch_2_b = SepConv(num_outputs_2_0a, num_outputs_2_0b, [3, 3, 3], padding=1)
        self.branch_3_a = nn.MaxPool3d([3, 3, 3], stride=1, padding=1) # Why give stride & padding here? Original code says no stride and padding="VALID" (hence none)
        self.branch_3_b = nn.Conv3d(input_dim, num_outputs_3_0b, [1, 1, 1])
        self.output_dim = num_outputs_0_0a + num_outputs_1_0b + num_outputs_2_0b + num_outputs_3_0b

        if self.use_gating:
            self.branch_0_g = self.gating_fn(self.branch_0)
            self.branch_1_g = self.gating_fn(self.branch_1_b)
            self.branch_2_g = self.gating_fn(self.branch_2_b)
            self.branch_3_g = self.gating_fn(self.branch_3_b)

    def forward(self, input):
        b0 = self.branch_0(input)
        b1 = self.branch_1_a(input)
        b1 = self.branch_1_b(b1)
        b2 = self.branch_2_a(input)
        b2 = self.branch_2_b(b2)
        b3 = self.branch_3_a(input)
        b3 = self.branch_3_b(b3)
        if self.use_gating:
            b0 = self.branch_0_g(b0)
            b1 = self.branch_1_g(b1)
            b2 = self.branch_2_g(b2)
            b3 = self.branch_2_g(b3)
        return torch.cat((b0, b1, b2, b3), dim=1)

class SepConv(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            kernel_size,
            stride=1,
            padding=0
            ):
        super(SepConv, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
        temporal_kernel_size = [kernel_size[0], 1, 1]
        if isinstance(stride, list) and len(stride) == 3:
            spatial_stride = [1, stride[1], stride[2]]
            temporal_stride = [stride[0], 1, 1]
        else:
            spatial_stride = [1, stride, stride]
            temporal_stride = [stride, 1, 1]
        if isinstance(padding, list) and len(padding) == 3:
            spatial_padding = [0, padding[1], padding[2]]
            temporal_padding = [padding[0], 0, 0]
        else:
            spatial_padding = [0, padding, padding]
            temporal_padding = [padding, 0, 0]
        
        self.conv1 = nn.Conv3d(input_dim, output_dim, spatial_kernel_size,
                               spatial_stride, spatial_padding, bias=False)
        # Batch norm?
        self.conv2 = nn.Conv3d(output_dim, output_dim, temporal_kernel_size,
                               temporal_stride, temporal_padding, bias=False)
        # Batch norm?

    def forward(self, input):
        out = self.relu(self.conv1(input))
        out = self.relu(self.conv2(input))
        return out
    
class MaxPool3dTFPadding(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super(MaxPool3dTFPadding, self).__init__()
        padding_shape = self._get_padding_shape(kernel_size, stride)
        self.padding_shape = padding_shape
        self.pad = nn.ConstantPad3d(padding_shape, 0)
        self.pool = nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def _get_padding_shape(self, kernel_size, stride):
        def _pad_to_bottom(filter_dim, stride_val):
            pad_along = max(filter_dim - stride_val, 0)
            pad_top = pad_along // 2
            pad_bottom = pad_along - pad_top
            return pad_top, pad_bottom
        
        padding_shape = []
        for filter_dim, stride_val in zip(kernel_size, stride):
            pad_top, pad_bottom = _pad_to_bottom(filter_dim, stride_val)
            padding_shape.append(pad_top)
            padding_shape.append(pad_bottom)
        depth_top = padding_shape.pop(0)
        depth_bottom = padding_shape.pop(0)
        padding_shape.append(depth_top)
        padding_shape.append(depth_bottom)
        return tuple(padding_shape)
    
    def forward(self, input):
        out = self.pad(input)
        out = self.pool(out)
        return out

class S3D(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.1, gating=False):
        super(S3D, self).__init__()
        self.num_classes = num_classes
        self.gating = gating
        # B x 64 x 224 x 224 x 3
        self.conv2d_1a = SepConv(3, 64, [3, 7, 7], stride=2, padding=[1, 2, 2]) # why 3 if article says 7? Also, how to find the padding?
        self.maxpool_2a = MaxPool3dTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        # B x 32 x 112 x 112 x 64
        self.conv2d_2b = nn.Conv3d(64, 64, [1, 1, 1], stride=1)
        # B x 32 x 112 x 112 x 64
        self.conv2d_2c = SepConv(64, 192, [3, 3, 3], stride=1, padding=1)
        # B x 32 x 112 x 112 x 192
        self.maxpool_3a = MaxPool3dTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.mixed_3b = SepInc(192, 64, 96, 128, 16, 32, 32)
        self.mixed_3c = SepInc(self.mixed_3b.output_dim, 128, 128, 192, 32, 96, 64)
        self.maxpool_4a = MaxPool3dTFPadding(kernel_size=(3, 3, 3), stride=(2, 2, 2))
        self.mixed_4b = SepInc(self.mixed_3c.output_dim, 192, 96, 208, 16, 48, 64)
        self.mixed_4c = SepInc(self.mixed_4b.output_dim, 160, 112, 224, 24, 64, 64)
        self.mixed_4d = SepInc(self.mixed_4c.output_dim, 128, 128, 256, 24, 64, 64)
        self.mixed_4e = SepInc(self.mixed_4d.output_dim, 112, 144, 288, 32, 64, 64)
        self.mixed_4f = SepInc(self.mixed_4e.output_dim, 256, 160, 320, 32, 128, 128)
        self.maxpool_5a = MaxPool3dTFPadding(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.mixed_5b = SepInc(self.mixed_4f.output_dim, 256, 160, 320, 32, 128, 128)
        self.mixed_5c = SepInc(self.mixed_5b.output_dim, 384, 192, 384, 48, 128, 128)
        self.avgpool_0a = nn.AvgPool3d((2, 7, 7), stride=1)
        self.dropout_0b = nn.Dropout3d(p=dropout, inplace=True)
        self.conv2d_0b = nn.Conv3d(self.mixed_5c.output_dim, self.num_classes, [1, 1, 1], bias=False)

    def forward(self, input):
        net = self.conv2d_1a(input)
        net = self.maxpool_2a(net)
        net = self.conv2d_2b(net)
        net = self.conv2d_2c(net)
        net = self.maxpool_3a(net)
        net = self.mixed_3b(net)
        net = self.mixed_3c(net)
        net = self.maxpool_4a(net)
        net = self.mixed_4b(net)
        net = self.mixed_4c(net)
        net = self.mixed_4d(net)
        net = self.mixed_4e(net)
        net = self.mixed_4f(net)
        net = self.maxpool_5a(net)
        net = self.mixed_5b(net)
        net = self.mixed_5c(net)
        net = self.avgpool_0a(net)
        net = self.dropout_0b(net)
        net = self.conv2d_0b(net)
        net = torch.mean(net, dim=(2, 3, 4))