import torch
import torch.nn as nn
import torchvision.models as models


class UNetModel(nn.Module):
    """
    UNet с предобученным ResNet34 энкодером.
    Предобученные веса ImageNet полезны для RGB.
    Для LAB/HSV энкодер дообучается с нуля (lr_encoder выше).
    """

    class _TwoConvLayers(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
            )

        def forward(self, x):
            return self.model(x)

    class _DecoderBlock(nn.Module):
        def __init__(self, in_channels, skip_channels, out_channels):
            super().__init__()
            self.transpose = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
            self.block = UNetModel._TwoConvLayers(out_channels + skip_channels, out_channels)

        def forward(self, x, y=None):
            x = self.transpose(x)
            if y is not None:
                x = torch.cat([x, y], dim=1)
            return self.block(x)

    def __init__(self, num_classes=1, pretrained: bool = True):
        super().__init__()

        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet34(weights=weights)

        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.enc1 = resnet.layer1   # 64ch,  H/4
        self.enc2 = resnet.layer2   # 128ch, H/8
        self.enc3 = resnet.layer3   # 256ch, H/16
        self.enc4 = resnet.layer4   # 512ch, H/32

        #                          in   skip  out
        self.dec_block1 = self._DecoderBlock(512, 256, 256)
        self.dec_block2 = self._DecoderBlock(256, 128, 128)
        self.dec_block3 = self._DecoderBlock(128,  64,  64)
        self.dec_block4 = self._DecoderBlock( 64,  64,  32)
        self.dec_block5 = self._DecoderBlock( 32,   0,  16)

        self.out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        x = self.dec_block1(e4, e3)
        x = self.dec_block2(x,  e2)
        x = self.dec_block3(x,  e1)
        x = self.dec_block4(x,  e0)
        x = self.dec_block5(x)

        return self.out(x)

    def get_param_groups(self, lr_encoder: float, lr_decoder: float, weight_decay: float):
        """Разные lr для энкодера (предобучен) и декодера (с нуля)"""
        return [
            {"params": self.enc0.parameters(), "lr": lr_encoder},
            {"params": self.enc1.parameters(), "lr": lr_encoder},
            {"params": self.enc2.parameters(), "lr": lr_encoder},
            {"params": self.enc3.parameters(), "lr": lr_encoder},
            {"params": self.enc4.parameters(), "lr": lr_encoder},
            {"params": self.dec_block1.parameters(), "lr": lr_decoder},
            {"params": self.dec_block2.parameters(), "lr": lr_decoder},
            {"params": self.dec_block3.parameters(), "lr": lr_decoder},
            {"params": self.dec_block4.parameters(), "lr": lr_decoder},
            {"params": self.dec_block5.parameters(), "lr": lr_decoder},
            {"params": self.out.parameters(),        "lr": lr_decoder},
        ]
