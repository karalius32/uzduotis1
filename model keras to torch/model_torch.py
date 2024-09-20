import torch 
from torch import nn


# Final model
class Model(nn.Module):
    def __init__(self, in_channels, num_of_classes, filters=8, depth=3, activation="relu", activation_args=None):
        super().__init__()
        self.depth = depth
        # Encoder
        self.encoder1 = EncodingBlock(in_channels=in_channels, out_channels=filters, kernel_size=5, max_pool=True, max_pool_size=2)
        self.encoder_res_blocks = [
            EncodingResBlock(in_channels=filters * (2 ** i), out_channels=filters * (2 ** (i + 1)), kernel_size=3, max_pool=True, max_pool_size=2, batch_norm=True, activation=activation, activation_args=activation_args)
            for i in range(depth)
        ]
        self.encoder2 = EncodingResBlock(in_channels=filters * (2 ** (depth)), out_channels=filters * (2 ** (depth + 1)), kernel_size=3, max_pool=False, batch_norm=True, activation=activation, activation_args=activation_args)
        # Head
        self.aspp = ASPP(in_channels=filters * (2 ** (depth + 1)), out_channels=filters * (2 ** (depth + 1)), kernel_size=3, depth=3, activation=activation, activation_args=activation_args)
        # Decoder
        self.decoder_res_blocks = [
            DecodingResBlock(in_channels=filters * (2 ** (depth + 1)), out_channels=filters * (2 ** (depth - i)), kernel_size=3, up_sample_size=2, batch_norm=True, activation=activation, activation_args=activation_args)
            for i in range(depth)
        ]
        self.decoder1 = DecodingResBlock(in_channels=filters * 2, out_channels=filters, kernel_size=3, up_sample_size=2, batch_norm=True, activation=activation, activation_args=activation_args)
        # Outputs
        final_activation = "softmax" if num_of_classes > 1 else "sigmoid"
        conv_block = ConvBlock(in_channels=filters, out_channels=num_of_classes * 2, kernel_size=3, batch_norm=True, activation=activation, activation_args=activation_args)
        final_conv = nn.Conv2d(in_channels=num_of_classes * 2, out_channels=num_of_classes, kernel_size=1)
        nn.init.xavier_normal(final_conv.weight)
        self.final_seq = nn.Sequential(
            conv_block,
            final_conv,
            ActivationBlock(activation=final_activation, activation_args=None)
        )


    def forward(self, input):
        skip_connections = []
        # Encoder
        enc_res, enc = self.encoder1(input)
        skip_connections.append(enc_res)
        for i in range(self.depth):
            enc_res, enc = self.encoder_res_blocks[i](enc)
            skip_connections.append(enc_res)
        _, bottleneck =  self.encoder2(enc)
        # Head
        bottleneck = self.aspp(bottleneck)
        # Decoder
        dec = bottleneck
        for i in range(self.depth):
            dec = self.decoder_res_blocks[i](dec)
        dec = self.decoder1(dec)
        # Output
        output = self.final_seq(dec)
        return output


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
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = ActivationBlock(activation=activation, activation_args=activation_args)
        self.do_batch_norm = batch_norm

    def forward(self, input):
        flow = self.zero_pad(input)
        flow = self.conv(flow)
        if self.do_batch_norm:
            flow = self.batch_norm(flow)
        flow = self.activation(flow)
        return flow


class AtrousConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, activation, activation_args):
        super().__init__()
        self.atrous_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation)
        nn.init.kaiming_normal(self.atrous_conv.weight)
        self.seq = nn.Sequential(
            nn.ZeroPad2d(padding=dilation * (kernel_size // 2)),
            self.atrous_conv,
            nn.BatchNorm2d(num_features=out_channels),
            ActivationBlock(activation=activation, activation_args=activation_args)
        )
    
    def forward(self, input):
        return self.seq(input)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, batch_norm=True, activation=None, activation_args=None):
        super().__init__()
        self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.kaiming_normal(self.shortcut_conv.weight)
        self.conv_block_seq = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, batch_norm=batch_norm, activation=activation, activation_args=activation_args),
            ConvBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, batch_norm=batch_norm)
        )
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
        self.conv_block_seq = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, batch_norm=batch_norm, activation=activation, activation_args=activation_args),
            ConvBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, batch_norm=batch_norm, activation=activation, activation_args=activation_args)
        )
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
        self.upsample_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(size=up_sample_size),
            ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, batch_norm=batch_norm, activation=activation, activation_args=activation_args)
        )
        self.res_block = ResBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, batch_norm=batch_norm, activation=activation, activation_args=activation_args)
    
    def forward(self, input, skip_connection):
        flow = self.upsample_conv(input)
        flow = torch.concatenate((flow, skip_connection), axis=1)
        return flow


# Atrous spatial pyramid pool block
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=8, kernel_size=3, depth=3, activation="relu", activation_args=None):
        super().__init__()
        self.atrous_conv_blocks = []
        for i in range(depth):
            dilation = 2 ** i
            self.atrous_conv_blocks.append(AtrousConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, activation=activation, activation_args=activation_args))
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv_block = ConvBlock(in_channels=depth*out_channels+in_channels, out_channels=out_channels, kernel_size=1, batch_norm=True, activation=activation, activation_args=activation_args)

    def forward(self, input):
        atrous_conv_outputs = [atrous_conv(input) for atrous_conv in self.atrous_conv_blocks]
        pool = self.pooling(input)
        pool = nn.functional.upsample_bilinear(input=pool, size=atrous_conv_outputs[0][-2:])
        flow = torch.concatenate([*atrous_conv_outputs, pool])
        flow = self.conv_block(flow)
        return flow