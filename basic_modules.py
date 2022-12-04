from torch import nn
import torch


class double_conv(nn.Module):
    """
    (conv + BatchNorm + ReLU) * 2
    """

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    """
    input conv
    only changes the number of channels
    """

    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    """
    downsample
    """

    def __init__(self, in_ch, out_ch, type='max'):
        super(down, self).__init__()
        if type == 'conv':
            self.mpconv = nn.Sequential(
                #nn.MaxPool2d(kernel_size=2),
                #double_conv(in_ch, out_ch)

                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1),
                double_conv(out_ch, out_ch),
            )
        elif type == 'max':
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(kernel_size=2),
                double_conv(in_ch, out_ch)

                #nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1),
                #double_conv(out_ch, out_ch),
            )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    """
    upsample
    """

    def __init__(self, in_ch, out_ch, bilinear=False, op="none"):
        super(up, self).__init__()
        self.bilinear = bilinear
        self.op = op
        if self.bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_ch, in_ch // 2, 1), )
        else:
            self.up = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1,
                                         output_padding=1)
        assert op in ["concat", "none"]

        if op == "concat":
            self.conv = double_conv(in_ch, out_ch)
        else:
            self.conv = double_conv(out_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)

        if self.op == "concat":
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1

        x = self.conv(x)
        return x


class outconv(nn.Module):
    """
    output 1x1 conv
    only changes the number of channels
    """

    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x