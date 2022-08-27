import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""norm layer"""

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        var = x.view(x.size(0), -1).var(1).view(*shape)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


"""basic block"""


class ConvBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='reflect', is3d=False):
        super().__init__()
        self.use_bias = False

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm3d(norm_dim, affine=True) if is3d else nn.BatchNorm2d(norm_dim, affine=True)
        elif norm == 'in':
            self.norm = nn.InstanceNorm3d(norm_dim, affine=True) if is3d else nn.InstanceNorm2d(norm_dim, affine=True)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim, affine=True)
            self.use_bias = True
        elif norm == 'none':
            self.norm = None
            self.use_bias = True
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        conv_fn = nn.Conv3d if is3d else nn.Conv2d
        self.conv = conv_fn(input_dim, output_dim, kernel_size, stride, padding, bias=self.use_bias, padding_mode=pad_type)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='reflect', is3d=False):
        super(ResBlock, self).__init__()

        model = []
        model += [ConvBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type, is3d=is3d)]
        model += [ConvBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type, is3d=is3d)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='reflect', is3d=False):
        super(ResBlocks, self).__init__()
        self.model = []
        for _ in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, is3d=is3d)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = False

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim, affine=True)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim, affine=True)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim, affine=True)
            use_bias = True
        elif norm == 'none':
            self.norm = None
            use_bias = True
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for _ in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))