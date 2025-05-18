import torch
import torch.nn as nn

class AttentionBlock3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
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
            nn.Conv3d(F_int + 1, 1, kernel_size=1),  # 1 extra channel for template map
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x, template_map):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)

        # Resize template_map to match psi shape and concatenate
        if template_map.shape[2:] != psi.shape[2:]:
            template_map = nn.functional.interpolate(template_map, size=psi.shape[2:], mode='trilinear', align_corners=False)
        psi = torch.cat([psi, template_map], dim=1)

        psi = self.psi(psi)
        return x * psi
    
    
class AttentionUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super(AttentionUNet3D, self).__init__()

        self.encoder1 = self._block(in_channels, features[0])
        self.pool1 = nn.MaxPool3d(2)

        self.encoder2 = self._block(features[0], features[1])
        self.pool2 = nn.MaxPool3d(2)

        self.encoder3 = self._block(features[1], features[2])
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = self._block(features[2], features[3])

        self.upconv3 = nn.ConvTranspose3d(features[3], features[2], kernel_size=2, stride=2)
        self.att3 = AttentionBlock3D(F_g=features[2], F_l=features[2], F_int=features[1])
        self.decoder3 = self._block(features[3], features[2])

        self.upconv2 = nn.ConvTranspose3d(features[2], features[1], kernel_size=2, stride=2)
        self.att2 = AttentionBlock3D(F_g=features[1], F_l=features[1], F_int=features[0])
        self.decoder2 = self._block(features[2], features[1])

        self.upconv1 = nn.ConvTranspose3d(features[1], features[0], kernel_size=2, stride=2)
        self.att1 = AttentionBlock3D(F_g=features[0], F_l=features[0], F_int=features[0]//2)
        self.decoder1 = self._block(features[1], features[0])

        self.conv_out = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, t1c_input, template_map):
        enc1 = self.encoder1(t1c_input)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.upconv3(bottleneck)
        enc3 = self.att3(g=dec3, x=enc3, template_map=template_map)
        dec3 = self.decoder3(torch.cat((dec3, enc3), dim=1))

        dec2 = self.upconv2(dec3)
        enc2 = self.att2(g=dec2, x=enc2, template_map=template_map)
        dec2 = self.decoder2(torch.cat((dec2, enc2), dim=1))

        dec1 = self.upconv1(dec2)
        enc1 = self.att1(g=dec1, x=enc1, template_map=template_map)
        dec1 = self.decoder1(torch.cat((dec1, enc1), dim=1))

        return self.conv_out(dec1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )