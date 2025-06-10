import torch
import torch.nn as nn

import Model.constants as constants

from Helpers.tensor_helper import crop_or_pad

class AttentionBlock3D(nn.Module):
    '''
    An attention block that can be applied to the U-Net.
    '''
    def __init__(self, F_g, F_l, F_int, template_channels=constants.ATTENTION_BLOCK_TEMPLATE_CHANNELS):
        '''
        The constructor for the attention block.

        Parameters:
        - F_g(Int): channels of the gating signal.
        - F_l(Int): channels of the local feature.
        - F_int(Int): Intermediate channel size for attention computation.
        - template_channels(Int): channels in the external map.
        '''
        super(AttentionBlock3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int + template_channels, 1, kernel_size=1),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

        
    def forward(self, g, x, template_map):
        '''
        Implements the forward pass logic.

        Parameters:
        - g(nn.Sequential): Gating signal from decoder (coarser resolution).
        - x(nn.Sequential): Skip connection input from encoder (fine resolution).
        - template_map(torch.Tensor): Optional template providing external spatial guidance.
        '''
        g1 = self.W_g(g)
        x1 = self.W_x(x)
    
        if x1.shape[2:] != g1.shape[2:]:
            x1 = crop_or_pad(x1, g1.shape[2:])
    
        psi = self.relu(g1 + x1)
    
        if template_map.shape[2:] != psi.shape[2:]:
            template_map = nn.functional.interpolate(template_map, size=psi.shape[2:], mode='trilinear', align_corners=False)
    
        psi = torch.cat([psi, template_map], dim=1)
        psi = self.psi(psi)
    
        psi = crop_or_pad(psi, x.shape[2:])
    
        return x * psi

class AttentionUNet3D(nn.Module):
    '''
    An implementation of a U-Net with attention.
    '''
    def __init__(self, in_channels, out_channels, features=constants.ATTENTION_U_NET_FEATURES_PER_LAYER):
        '''
        The constructor for the U-Net

        Parameters:
        - in_channels(Int): number of input channels.
        - out_channels(Int): number of output channels.
        - features([Int]): number of features at each level.
        '''
        super(AttentionUNet3D, self).__init__()
        self.encoder1 = self.encode_decode_layer(in_channels, features[0])
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = self.encode_decode_layer(features[0], features[1])
        self.pool2 = nn.MaxPool3d(2)
        self.encoder3 = self.encode_decode_layer(features[1], features[2])
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = self.encode_decode_layer(features[2], features[3])
        self.upconv3 = nn.ConvTranspose3d(features[3], features[2], 
                                          kernel_size=constants.ATTENTION_U_NET_TRANSPOSED_CONVOLUTION_KERNEL_SIZE, 
                                          stride=constants.ATTENTION_U_NET_TRANSPOSED_CONVOLUTION_PADDING_SIZE)
        self.att3 = AttentionBlock3D(features[2], features[2], features[1])
        self.decoder3 = self.encode_decode_layer(features[3], features[2])
        self.upconv2 = nn.ConvTranspose3d(features[2], features[1],                                          
                                          kernel_size=constants.ATTENTION_U_NET_TRANSPOSED_CONVOLUTION_KERNEL_SIZE, 
                                          stride=constants.ATTENTION_U_NET_TRANSPOSED_CONVOLUTION_PADDING_SIZE)
        self.att2 = AttentionBlock3D(features[1], features[1], features[0])
        self.decoder2 = self.encode_decode_layer(features[2], features[1])
        self.upconv1 = nn.ConvTranspose3d(features[1], features[0],
                                          kernel_size=constants.ATTENTION_U_NET_TRANSPOSED_CONVOLUTION_KERNEL_SIZE, 
                                          stride=constants.ATTENTION_U_NET_TRANSPOSED_CONVOLUTION_PADDING_SIZE)
        self.att1 = AttentionBlock3D(features[0], features[0], features[0]//2)
        self.decoder1 = self.encode_decode_layer(features[1], features[0])
        self.conv_out = nn.Conv3d(features[0], out_channels, kernel_size=constants.ATTENTION_U_NET_OUTPUT_CONVOLUTION_KERNEL_SIZE)

    def encode_decode_layer(self, in_channels, out_channels):
        '''
        Creates an encoder/decoder layer using the given channels.

        Parameters:
        - in_channels(Int): number of input channels.
        - out_channels(Int): number of output channels.

        Returns:
        - nn.Sequential: a pytorch layer.
        '''
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=constants.ATTENTION_U_NET_CONVOLUTION_KERNEL_SIZE,
                      padding=constants.ATTENTION_U_NET_CONVOLUTION_PADDING_SIZE),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=constants.ATTENTION_U_NET_CONVOLUTION_KERNEL_SIZE,
                      padding=constants.ATTENTION_U_NET_CONVOLUTION_PADDING_SIZE),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_image, template_map):
        '''
        Implements the forward pass logic.

        Parameters:
        - input_image(torch.tensor): input image.
        - template_map(torch.tensor): the attention template.

        Returns:
        - Float: the output of the layer.
        '''
        enc1 = self.encoder1(input_image)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.upconv3(bottleneck)
        enc3 = self.att3(g=dec3, x=enc3, template_map=template_map)
        if enc3.shape[2:] != dec3.shape[2:]:
            diffD = enc3.size(2) - dec3.size(2)
            diffH = enc3.size(3) - dec3.size(3)
            diffW = enc3.size(4) - dec3.size(4)
            enc3 = enc3[:, :, diffD//2:diffD//2 + dec3.size(2),
                              diffH//2:diffH//2 + dec3.size(3),
                              diffW//2:diffW//2 + dec3.size(4)]
        dec3 = self.decoder3(torch.cat((dec3, enc3), dim=1))

        dec2 = self.upconv2(dec3)
        enc2 = self.att2(g=dec2, x=enc2, template_map=template_map)
        if enc2.shape[2:] != dec2.shape[2:]:
            diffD = enc2.size(2) - dec2.size(2)
            diffH = enc2.size(3) - dec2.size(3)
            diffW = enc2.size(4) - dec2.size(4)
            enc2 = enc2[:, :, diffD//2:diffD//2 + dec2.size(2),
                              diffH//2:diffH//2 + dec2.size(3),
                              diffW//2:diffW//2 + dec2.size(4)]
        dec2 = self.decoder2(torch.cat((dec2, enc2), dim=1))

        dec1 = self.upconv1(dec2)
        enc1 = self.att1(g=dec1, x=enc1, template_map=template_map)
        if enc1.shape[2:] != dec1.shape[2:]:
            diffD = enc1.size(2) - dec1.size(2)
            diffH = enc1.size(3) - dec1.size(3)
            diffW = enc1.size(4) - dec1.size(4)
            enc1 = enc1[:, :, diffD//2:diffD//2 + dec1.size(2),
                              diffH//2:diffH//2 + dec1.size(3),
                              diffW//2:diffW//2 + dec1.size(4)]
        dec1 = self.decoder1(torch.cat((dec1, enc1), dim=1))

        return self.conv_out(dec1)