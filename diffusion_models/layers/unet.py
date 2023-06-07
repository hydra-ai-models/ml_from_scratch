# Implementation of the UNet architecture with cross attention.

import torch
from torch import nn

class ResidualBlock2d(nn.Module):
    '''
        Implementation of a two dimensional residual block implementing the logic
        output = [Conv2d -> BatchNorm -> ReLU -> Conv2d -> Batch Norm](input) + input.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, mid_channels = None) -> None:
        '''
            Initializer function.

            Args:
                1. in_channels: Number of channels in the input tensor.
                2. out_channels: Number of channels in the output tensor.
                3. kernel_size: Size of the convolutional kernel to use for the Conv2d operations.
                4. mid_channels: Number of channels in the output of the first convolutional layer. If not specified,
                    this is set to out_channels.

            Returns:
                None
        '''
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels if mid_channels else out_channels

        self.layers = nn.Sequential(
                nn.Conv2d(self.in_channels, self.mid_channels, kernel_size, padding = 'same'),
                nn.BatchNorm2d(self.mid_channels),
                nn.ReLU(),
                nn.Conv2d(self.mid_channels, self.out_channels, kernel_size, padding = 'same'),
                nn.BatchNorm2d(self.out_channels))

        # Use 1x1 convolution before residual addition if input and output have different depths.
        self.residual_layer = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        residual = self.residual_layer(x)
        return residual + self.layers(x)

class UNetWithCrossAttention(nn.Module):
    '''
        UNet architecture with cross attention between input image and additional input 1d signal.
    '''
    def __init__(self, skip_connections:bool = False):
        '''
            Initializer function.

            Args:
                1. skip_connections: Whether to apply skip connections from input to output.

            Returns:
                None
        '''
        super().__init__()
        self.skip_connections = skip_connections

        # Encoder parameters.
        self.encoder_depths = [64, 128, 256, 512]
        self.encoder_strides = [1, 2, 2, 2]

        # Decoder parameters.
        self.decoder_depths = [512, 256, 128, 64]
        self.decoder_strides = [2, 2, 2, 1]

        self.input_depth = 3
        self.kernel_size = 3

        # Initial convolution to increase the depth of input before applying encoder.
        self.initial_image_conv = nn.Conv2d(self.input_depth, self.encoder_depths[0], self.kernel_size, padding = 'same')

        # Encoder layers.
        encoder_blocks_list = []
        for i in range(len(self.encoder_depths) - 1):
            encoder_blocks_list.extend([
                ResidualBlock2d(self.encoder_depths[i], self.encoder_depths[i], kernel_size = self.kernel_size),
                ResidualBlock2d(self.encoder_depths[i], self.encoder_depths[i + 1], kernel_size = self.kernel_size),
                nn.MaxPool2d(stride = self.encoder_strides[i + 1], kernel_size = self.kernel_size, padding = 1)
            ])
        self.encoder_blocks = nn.ModuleList(encoder_blocks_list)

        # Decoder layers.
        decoder_blocks_list = []
        for i in range(len(self.decoder_depths) - 1):
            skip_connection_multiplier = 2 if i > 0 and self.skip_connections else 1
            decoder_blocks_list.extend([
                ResidualBlock2d(self.decoder_depths[i] * skip_connection_multiplier, self.decoder_depths[i], kernel_size = self.kernel_size),
                ResidualBlock2d(self.decoder_depths[i], self.decoder_depths[i + 1], kernel_size = self.kernel_size),
                nn.ConvTranspose2d(self.decoder_depths[i + 1], self.decoder_depths[i + 1], stride = self.decoder_strides[i], kernel_size = self.kernel_size, padding = 1, output_padding = 1)
            ])
        self.decoder_blocks = nn.ModuleList(decoder_blocks_list)

        # Final convolution layer to take the encoder output to channel network output.
        self.final_image_conv = nn.Conv2d(self.decoder_depths[-1], self.input_depth, self.kernel_size, padding = 'same')

    def forward(self, input_image, input_text):
        [batch_size, current_input_depth, _, _] = input_image.shape
        assert current_input_depth == self.input_depth, f'Unexpected input depth {current_input_depth}. Expected {self.input_depth}.'

        features = self.initial_image_conv(input_image)
        encoder_features = []
        for (encoder_block_index, encoder_block) in enumerate(self.encoder_blocks):
            features = encoder_block(features)
            if encoder_block_index % 3 == 2:
                encoder_features.append(features)

        for (decoder_block_index, decoder_block) in enumerate(self.decoder_blocks):
            if decoder_block_index > 2 and decoder_block_index % 3 == 0:
                features = torch.cat([features, encoder_features[-1 * (decoder_block_index//3) - 1]], dim = 1)
            features = decoder_block(features)
        predictions = self.final_image_conv(features)
        return predictions
