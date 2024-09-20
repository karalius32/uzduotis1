import torch 
from torch import nn


# Basic blocks
class ActivationBlock(nn.Module):
    def __init__(self, activation=None, activation_args=None):
        super().__init__()
        match activation:
            case "leaky_relu":
                self.activation = nn.LeakyReLU(negative_slope=activation_args['negative_slope']) # in keras - alpha
            case "relu":
                self.activation = nn.ReLU()
            case "sigmoid":
                self.activation = nn.Sigmoid()
            case "softmax":
                self.activation = nn.Softmax()
            case None:
                self.activation = None

    def forward(self, input):
        if self.activation is None:
            return input
        return self.activation(input)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, batch_norm=True, activation=None, activation_args=None):
        super().__init__()
        self.zero_pad = nn.ZeroPad2d(padding=(kernel_size // 2, kernel_size // 2))
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        nn.init.kaiming_normal(self.conv.weight)
        self.batch_norm = nn.BatchNorm2d()
        self.activation = ActivationBlock(activation=activation, activation_args=activation_args)
        self.do_batch_norm = batch_norm

    def forward(self, input):
        flow = self.zero_pad(input)
        flow = self.conv(flow)
        if self.do_batch_norm:
            flow = self.batch_norm(flow)
        flow = self.activation(flow)
        return flow


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, batch_norm=True, activation=None, activation_args=None):
        super().__init__()
        self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.kaiming_normal(self.conv.weight)
        self.conv_block_seq = nn.Sequential([
            ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, batch_norm=batch_norm, activation=activation, activation_args=activation_args),
            ConvBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, batch_norm=batch_norm)
        ])
        self.activation = ActivationBlock(activation=activation, activation_args=activation_args)
    
    def forward(self, input):
        shortcut = self.shortcut_conv(input)
        flow = self.conv_block_seq(input)
        flow += shortcut
        flow = self.activation(flow)
        return flow


# Encoder / decoder blocks
class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels=8, kernel_size=3, max_pool=True, max_pool_size=2, batch_norm=True, activation="relu", activation_args=None):
        super().__init__()
        self.conv_block_seq = nn.Sequential([
            ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, batch_norm=batch_norm, activation=activation, activation_args=activation_args),
            ConvBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, batch_norm=batch_norm, activation=activation, activation_args=activation_args)
        ])
        self.max_pool = nn.MaxPool2d(kernel_size=max_pool_size)
        self.do_max_pool = max_pool
    
    def forward(self, input):
        flow = self.conv_block_seq(input)
        res = flow
        if self.do_max_pool:
            flow = self.max_pool(flow)
        return res, flow


class EncodingResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=8, kernel_size=3, max_pool=True, max_pool_size=2, batch_norm=True, activation="relu", activation_args=None):
        super().__init__()
        self.res_block = ResBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, batch_norm=batch_norm, activation=activation, activation_args=activation_args)
        self.max_pool = nn.MaxPool2d(kernel_size=max_pool_size)
        self.do_max_pool = max_pool

    def forward(self, input):
        flow = self.res_block(input)
        res = flow
        if self.do_max_pool:
            flow = self.max_pool(flow)
        return res, flow


class DecodingResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=8, kernel_size=3, up_sample_size=2, batch_norm=True, activation="relu", activation_args=None):
        super().__init__()
        self.upsample_conv = nn.Sequential([
            nn.UpsamplingBilinear2d(size=up_sample_size),
            ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, batch_norm=batch_norm, activation=activation, activation_args=activation_args)
        ])
        self.res_block = ResBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, batch_norm=batch_norm, activation=activation, activation_args=activation_args)
    
    def forward(self, input, skip_connection):
        flow = self.upsample_conv(input)
        flow = torch.concatenate((flow, skip_connection), axis=1)
        return flow


# Atrous spatial pyramid pool block
class ASPP(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward():
        pass