from functools import partial

import torch
from torch import nn as nn
from torch.nn import functional as F
from kornia.filters.kernels import get_spatial_gradient_kernel2d, get_spatial_gradient_kernel3d, normalize_kernel2d, get_sobel_kernel_3x3
import numpy as np
import scipy.ndimage as ndimage
import kornia


def conv3d(in_channels, out_channels, kernel_size, bias, padding):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple):
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='gcr', num_groups=8, padding=1):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='gcr', num_groups=8, padding=1):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups,
                                   padding=padding))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups,
                                   padding=padding))


class Attention_SE_CA(nn.Module):
    def __init__(self, channel):
        super(Attention_SE_CA, self).__init__()
        self.Global_Pool = nn.AdaptiveAvgPool3d(1)
        self.FC1 = nn.Sequential(nn.Linear(channel, channel),
                                 nn.ReLU(), )
        self.FC2 = nn.Sequential(nn.Linear(channel, channel),
                                 nn.Sigmoid(), )

    def forward(self, x):
        G = self.Global_Pool(x)
        G = G.view(G.size(0), -1)
        fc1 = self.FC1(G)
        fc2 = self.FC2(fc1)
        fc2 = torch.unsqueeze(fc2, 2)
        fc2 = torch.unsqueeze(fc2, 3)
        fc2 = torch.unsqueeze(fc2, 4)
        return fc2*x


# class SobelFilter(nn.Module):
#     def __init__(self):
#         super(SobelFilter, self).__init__()
#
#     def _sobel(self, x):
#         bs, channel, depth, height, width = x.shape
#         # print(f'x.shape = {x.shape}, type = {type(x)}')
#         edge = np.zeros(bs, channel, depth, height, width)
#         edge = torch.zeros_like(x).cuda()
#         x = x.detach().cpu().numpy()
#         edge = edge.detach().cpu().numpy()
#         # print(f'edge.shape = {edge.shape}, x type = {type(x)}')
#         for b in range(bs):
#             for c in range(channel):
#                 x_slice = x[b, c, :, :, :]
#                 sx = ndimage.sobel(x_slice, axis=0, mode='constant')
#                 sy = ndimage.sobel(x_slice, axis=1, mode='constant')
#                 edge[b, c, :, :, :] = np.hypot(sx, sy)
#                 edge[b, c, :, :, :] = (edge[b, c, :, :, :] - np.min(edge[b, c, :, :, :])) / (np.max(edge[b, c, :, :, :]) - np.min(edge[b, c, :, :, :]))
#         edge = torch.from_numpy(edge).float().cuda()
#         # print(f'edge type = {type(edge)}')
#         return edge
#
#     def forward(self, x):

# def get_sobel_kernel_3d():
#     """ Return a [3,1,3,3,3] sobel kernel"""
#     return torch.tensor(
#         [
#             [[[-1, 0, 1],
#               [-2, 0, 2],
#               [-1, 0, 1]],
#
#              [[-2, 0, 2],
#               [-4, 0, 4],
#               [-2, 0, 2]],
#
#              [[-1, 0, 1],
#               [-2, 0, 2],
#               [-1, 0, 1]]],
#
#             [[[-1, -2, -1],
#               [0, 0, 0],
#               [1, 2, 1]],
#
#              [[-2, -4, -2],
#               [0, 0, 0],
#               [2, 4, 2]],
#
#              [[-1, -2, -1],
#               [0, 0, 0],
#               [1, 2, 1]]],
#
#             [[[-1, -2, -1],
#               [-2, -4, -2],
#               [-1, -2, -1]],
#
#              [[0, 0, 0],
#               [0, 0, 0],
#               [0, 0, 0]],
#
#              [[1, 2, 1],
#               [2, 4, 2],
#               [1, 2, 1]]]
#         ]).unsqueeze(1)
#
#
# def spacialGradient_3d(image):
#     """ Implementation of a sobel 3d spatial gradient inspired by the kornia library.
#
#     :param image: Tensor [B,1,H,W,D]
#     :return: Tensor [B,3,H,W,D]
#
#     :Example:
#     H,W,D = (50,75,100)
#     image = torch.zeros((H,W,D))
#     mX,mY,mZ = torch.meshgrid(torch.arange(H),
#                               torch.arange(W),
#                               torch.arange(D))
#
#     mask_rond = ((mX - H//2)**2 + (mY - W//2)**2).sqrt() < H//4
#     mask_carre = (mX > H//4) & (mX < 3*H//4) & (mZ > D//4) & (mZ < 3*D//4)
#     mask_diamand = ((mY - W//2).abs() + (mZ - D//2).abs()) < W//4
#     mask = mask_rond & mask_carre & mask_diamand
#     image[mask] = 1
#
#
#     grad_image = spacialGradient_3d(image[None,None])
#     """
#
#     # sobel kernel is not implemented for 3D images yet in kornia
#     # grad_image = SpatialGradient3d(mode='sobel')(image)
#     kernel = get_sobel_kernel_3d().to(image.device).to(image.dtype)
#     spatial_pad = [1,1,1,1,1,1]
#     image_padded = F.pad(image,spatial_pad,'replicate').repeat(1,3,1,1,1)
#     grad_image =  F.conv3d(image_padded,kernel,padding=0,groups=3,stride=1)
#
#     return grad_image

class SobelFilter(nn.Module):

    def __init__(self):
        super(SobelFilter, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def get_gray(self,x):
        '''
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        # x_list = []
        # for i in range(x.shape[1]):
        #     x_i = x[:, i]
        #     x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
        #     x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
        #     x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
        #     x_list.append(x_i)

        # x = torch.cat(x_list, dim=1)
        bs, channel, depth, heigt, width = x.shape
        for b in range(bs):
            for c in range(channel):
                x_slice = x[b, c, :, :, :]
        x_v = F.conv2d(x_slice, self.weight_v, padding=1)
        x_h = F.conv2d(x_slice, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x


def get_sobel_kernel_3d(self):
    """ Return a [3,1,3,3,3] sobel kernel"""
    return torch.tensor(
        [
            [[[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]],

             [[-2, 0, 2],
              [-4, 0, 4],
              [-2, 0, 2]],

             [[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]],

            [[[-1, -2, -1],
              [0, 0, 0],
              [1, 2, 1]],

             [[-2, -4, -2],
              [0, 0, 0],
              [2, 4, 2]],

             [[-1, -2, -1],
              [0, 0, 0],
              [1, 2, 1]]],

            [[[-1, -2, -1],
              [-2, -4, -2],
              [-1, -2, -1]],

             [[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]],

             [[1, 2, 1],
              [2, 4, 2],
              [1, 2, 1]]]
        ]).unsqueeze(1)


class ExtResNetBlock(nn.Module):
    """
    Basic UNet block consisting of a SingleConv followed by the residual block.
    The SingleConv takes care of increasing/decreasing the number of channels and also ensures that the number
    of output channels is compatible with the residual block that follows.
    This block can be used instead of standard DoubleConv in the Encoder module.
    Motivated by: https://arxiv.org/pdf/1706.00120.pdf
    Notice we use ELU instead of ReLU (order='cge') and put non-linearity after the groupnorm.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8, get_edge=False, **kwargs):
        super(ExtResNetBlock, self).__init__()

        # first convolution
        self.conv1 = SingleConv(in_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order,
                                num_groups=num_groups)

        # create non-linearity separately
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

        self.dropout3d = nn.Dropout3d(p=0.1)
        # self.attention = Attention_SE_CA(in_channels)
        # print(f'in_channels = {in_channels}, out_channels = {out_channels}')
        # self.sobel = SobelFilter()
        # self.sobel = kornia.filters.spatial_gradient3d()
        self.get_edge = get_edge
        self.edge_conv = nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))


    def _spatialGradient_3d(self, image):
        """ Implementation of a sobel 3d spatial gradient inspired by the kornia library.

        :param image: Tensor [B,1,H,W,D]
        :return: Tensor [B,3,H,W,D]

        :Example:
        H,W,D = (50,75,100)
        image = torch.zeros((H,W,D))
        mX,mY,mZ = torch.meshgrid(torch.arange(H),
                                  torch.arange(W),
                                  torch.arange(D))

        mask_rond = ((mX - H//2)**2 + (mY - W//2)**2).sqrt() < H//4
        mask_carre = (mX > H//4) & (mX < 3*H//4) & (mZ > D//4) & (mZ < 3*D//4)
        mask_diamand = ((mY - W//2).abs() + (mZ - D//2).abs()) < W//4
        mask = mask_rond & mask_carre & mask_diamand
        image[mask] = 1


        grad_image = spacialGradient_3d(image[None,None])
        """

        # sobel kernel is not implemented for 3D images yet in kornia
        # grad_image = SpatialGradient3d(mode='sobel')(image)
        kernel = get_sobel_kernel_3d(image).to(image.device).to(image.dtype)
        spatial_pad = [1, 1, 1, 1, 1, 1]
        image_padded = F.pad(image, spatial_pad, 'replicate').repeat(1, 3, 1, 1, 1)
        grad_image = F.conv3d(image_padded, kernel, padding=0, groups=3, stride=1)

        return grad_image

    def _sobel(self, x):
        bs, channel, depth, height, width = x.shape
        # print(f'x.shape = {x.shape}, type = {type(x)}')
        # edge = np.zeros((bs, channel, depth, height, width))
        edge = torch.zeros_like(x).cuda()
        x = x.detach().cpu().numpy()
        edge = edge.detach().cpu().numpy()
        # print(f'edge.shape = {edge.shape}, x type = {type(x)}')
        for b in range(bs):
            for c in range(channel):
                x_slice = x[b, c, :, :, :]
                sx = ndimage.sobel(x_slice, axis=0, mode='constant')
                sy = ndimage.sobel(x_slice, axis=1, mode='constant')
                edge[b, c, :, :, :] = np.hypot(sx, sy)
                edge[b, c, :, :, :] = (edge[b, c, :, :, :] - np.min(edge[b, c, :, :, :])) / (np.max(edge[b, c, :, :, :]) - np.min(edge[b, c, :, :, :]))
        edge = torch.from_numpy(edge).float().cuda()
        return edge

    def _sobel_torch(self, x):
        bs, channel, depth, height, width = x.shape
        # print(f'x.shape = {x.shape}, type = {type(x)}')
        # edge = np.zeros((bs, channel, depth, height, width))

        # edge = torch.zeros_like(x)
        edge = torch.zeros_like(x)

        # print(f'edge.shape = {edge.shape}, x type = {type(x)}')
        for d in range(depth):
            x_slice = x[:, :, d, :, :]
            sobel_slice = kornia.sobel(x_slice)
            edge[:, :, d, :, :] = sobel_slice
            # edge[b, c, :, :, :] = (edge[b, c, :, :, :] - np.min(edge[b, c, :, :, :])) / (np.max(edge[b, c, :, :, :]) - np.min(edge[b, c, :, :, :]))
        return edge

    def forward(self, x):
        # apply first convolution and save the output as a residual
        if self.get_edge:
            # print(f'1get_edge = {self.get_edge}')
            out = self.conv1(x)
            residual = out
            # print(f'out size = {out.size()}')
            edge = torch.sigmoid(out)
            edge = self._sobel_torch(edge)
            # edge = kornia.filters.spatial_gradient3d(edge)
            # edge = self._spatialGradient_3d(edge)
            # print(f'edge size = {edge.size()}')
            # residual block
            out = self.conv2(out)
            out = self.conv3(out)

            out += residual
            out = self.dropout3d(out)
            out = self.non_linearity(out)

            out = out * (1 + torch.sigmoid(edge))
        else:
            # print(f'2get_edge = {self.get_edge}')
            out = self.conv1(x)
            residual = out

            # residual block
            out = self.conv2(out)
            out = self.conv3(out)

            out += residual
            out = self.dropout3d(out)
            out = self.non_linearity(out)

        return out


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (int or tuple): the size of the window
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=2, pool_type='max', basic_module=DoubleConv, conv_layer_order='gcr',
                 num_groups=8, padding=1):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding,
                                         get_edge=True)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsampling layer
    (either learned ConvTranspose3d or nearest neighbor interpolation) followed by a basic module (DoubleConv or ExtResNetBlock).
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        upsample (boole): should the input be upsampled
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, scale_factor=(2, 2, 2), basic_module=DoubleConv,
                 conv_layer_order='gcr', num_groups=8, mode='nearest', padding=1, upsample=True):
        super(Decoder, self).__init__()

        if upsample:
            if basic_module == DoubleConv:
                # if DoubleConv is the basic_module use interpolation for upsampling and concatenation joining
                self.upsampling = InterpolateUpsampling(mode=mode)
                # concat joining
                self.joining = partial(self._joining, concat=True)
            else:
                # if basic_module=ExtResNetBlock use transposed convolution upsampling and summation joining
                self.upsampling = TransposeConvUpsampling(in_channels=in_channels, out_channels=out_channels,
                                                          kernel_size=conv_kernel_size, scale_factor=scale_factor)
                # sum joining
                self.joining = partial(self._joining, concat=False)
                # adapt the number of in_channels for the ExtResNetBlock
                in_channels = out_channels
        else:
            # no upsampling
            self.upsampling = NoUpsampling()
            # concat joining
            self.joining = partial(self._joining, concat=True)

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding,
                                         get_edge=False)

    def forward(self, encoder_features, x):
        """
        :param encoder_features:
        :param x:
        :return:
        encoder_features size = torch.Size([2, 512, 8, 8, 8]), x size = torch.Size([2, 1024, 4, 4, 4])
        encoder_features size = torch.Size([2, 256, 16, 16, 16]), x size = torch.Size([2, 512, 8, 8, 8])
        encoder_features size = torch.Size([2, 128, 32, 32, 32]), x size = torch.Size([2, 256, 16, 16, 16])
        encoder_features size = torch.Size([2, 64, 64, 64, 64]), x size = torch.Size([2, 128, 32, 32, 32])
        type of encoder_features, x : <class 'torch.Tensor'>
        """
        # print(f'1 encoder_features size = {encoder_features.size()}, x size = {x.size()}')

        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


class ASPP(nn.Module):
    """
    3D Astrous Spatial Pyramid Pooling
    Code modified from https://github.com/lvpeiqing/SAR-U-Net-liver-segmentation/blob/master/models/se_p_resunet/se_p_resunet.py
    """
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.pool = nn.MaxPool3d(3)
        self.aspp_block1 = nn.Sequential(
            nn.Conv3d(
                in_dims, out_dims, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(rate[0], rate[0], rate[0]),
                dilation=(rate[0], rate[0], rate[0])
            ),
            nn.PReLU(),
            # nn.BatchNorm3d(out_dims),
            nn.GroupNorm(8, out_dims)
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv3d(
                in_dims, out_dims, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(rate[1], rate[1], rate[1]),
                dilation=(rate[1], rate[1], rate[1])
            ),
            nn.PReLU(),
            # nn.BatchNorm3d(out_dims),
            nn.GroupNorm(8, out_dims)
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv3d(
                in_dims, out_dims, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(rate[2], rate[2], rate[2]),
                dilation=(rate[2], rate[2], rate[2])
            ),
            nn.PReLU(),
            # nn.BatchNorm3d(out_dims),
            nn.GroupNorm(8, out_dims)
        )

        self.output = nn.Conv3d(len(rate) * out_dims, out_dims, kernel_size=(1, 1, 1))
        self._init_weights()

    def forward(self, x):
        print(f'aspp start x size = {x.size()}')
        x = self.pool(x)
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        print(f'x1 = {x1.size()}, x2 = {x2.size()}, x3 = {x3.size()}')
        out = torch.cat([x1, x2, x3], dim=1)
        print(f'aspp end x size = {x.size()}')
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                    pool_kernel_size):
    # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        # print(f'building blocks: fmaps = {f_maps}, i = {i}, len = {len(f_maps)}')
        if i == 0:
            encoder = Encoder(in_channels, out_feature_num,
                              apply_pooling=False,  # skip pooling in the first encoder
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              padding=conv_padding)
        # elif i == len(f_maps) - 1:
        #     print(f'last layer of encoder with aspp block')
        #     encoder = ASPP(f_maps[i - 1], out_feature_num)
        else:
            # TODO: adapt for anisotropy in the data, i.e. use proper pooling kernel to make the data isotropic after 1-2 pooling operations
            encoder = Encoder(f_maps[i - 1], out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              pool_kernel_size=pool_kernel_size,
                              padding=conv_padding)

        encoders.append(encoder)

    return nn.ModuleList(encoders)


def create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups, upsample):
    # create decoder path consisting of the Decoder modules. The length of the decoder list is equal to `len(f_maps) - 1`
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        if basic_module == DoubleConv:
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
        else:
            in_feature_num = reversed_f_maps[i]

        out_feature_num = reversed_f_maps[i + 1]

        # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
        # currently strides with a constant stride: (2, 2, 2)

        _upsample = True
        if i == 0:
            # upsampling can be skipped only for the 1st decoder, afterwards it should always be present
            _upsample = upsample

        decoder = Decoder(in_feature_num, out_feature_num,
                          basic_module=basic_module,
                          conv_layer_order=layer_order,
                          conv_kernel_size=conv_kernel_size,
                          num_groups=num_groups,
                          padding=conv_padding,
                          upsample=_upsample)
        decoders.append(decoder)
    return nn.ModuleList(decoders)


class AbstractUpsampling(nn.Module):
    """
    Abstract class for upsampling. A given implementation should upsample a given 5D input tensor using either
    interpolation or learned transposed convolution.
    """

    def __init__(self, upsample):
        super(AbstractUpsampling, self).__init__()
        self.upsample = upsample

    def forward(self, encoder_features, x):
        # get the spatial dimensions of the output given the encoder_features
        output_size = encoder_features.size()[2:]
        # upsample the input and return
        return self.upsample(x, output_size)


class InterpolateUpsampling(AbstractUpsampling):
    """
    Args:
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
            used only if transposed_conv is False
    """

    def __init__(self, mode='nearest'):
        upsample = partial(self._interpolate, mode=mode)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


class TransposeConvUpsampling(AbstractUpsampling):
    """
    Args:
        in_channels (int): number of input channels for transposed conv
            used only if transposed_conv is True
        out_channels (int): number of output channels for transpose conv
            used only if transposed_conv is True
        kernel_size (int or tuple): size of the convolving kernel
            used only if transposed_conv is True
        scale_factor (int or tuple): stride of the convolution
            used only if transposed_conv is True
    """

    def __init__(self, in_channels=None, out_channels=None, kernel_size=3, scale_factor=(2, 2, 2)):
        # make sure that the output size reverses the MaxPool3d from the corresponding encoder
        upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor,
                                      padding=1)
        super().__init__(upsample)


class NoUpsampling(AbstractUpsampling):
    def __init__(self):
        super().__init__(self._no_upsampling)

    @staticmethod
    def _no_upsampling(x, size):
        return x
