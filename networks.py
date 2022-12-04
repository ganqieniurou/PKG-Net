from basic_modules import *


class Encoder_1(nn.Module):
    """
    encoder for k=1
    """
    def __init__(self, in_size=12, fea_size=64):
        super(Encoder_1, self).__init__()
        self.in_conv = inconv(in_size, fea_size)
        self.down1 = down(fea_size, fea_size * 2, 'conv')
        self.down2 = down(fea_size * 2, fea_size * 4, 'conv')
        self.down3 = down(fea_size * 4, fea_size * 8, 'conv')

    def forward(self, x):
        fea32 = self.in_conv(x)
        fea16 = self.down1(fea32)
        fea8 = self.down2(fea16)
        fea4 = self.down3(fea8)
        return fea4, fea8, fea16, fea32


class Decoder_1(nn.Module):
    """
    decoder for k=1
    """
    def __init__(self, fea_num=64):
        super(Decoder_1, self).__init__()
        self.up1 = up(fea_num * 8, fea_num * 4, op='concat')
        self.up2 = up(fea_num * 4, fea_num * 2, op='concat')
        self.up3 = up(fea_num * 2, fea_num, op='concat')
        self.outconv = outconv(fea_num, 3)

    def forward(self, fea4, skip8, skip16, skip32):
        fea8 = self.up1(fea4, skip8)
        fea16 = self.up2(fea8, skip16)
        fea32 = self.up3(fea16, skip32)
        out = self.outconv(fea32)
        return out


class Encoder_2(nn.Module):
    """
    encoder for k=2
    """
    def __init__(self, in_size=12, fea_size=64):
        super(Encoder_2, self).__init__()
        self.in_conv = inconv(in_size, fea_size)
        self.down1 = down(fea_size, fea_size * 2)
        self.down2 = down(fea_size * 2, fea_size * 4)
        self.down3 = down(fea_size * 4, fea_size * 8)

    def forward(self, x):
        fea32 = self.in_conv(x)
        fea16 = self.down1(fea32)
        fea8 = self.down2(fea16)
        fea4 = self.down3(fea8)
        return fea4, fea16, fea32


class Decoder_2(nn.Module):
    """
    decoder for k=2
    """
    def __init__(self, fea_num=64):
        super(Decoder_2, self).__init__()
        self.up1 = up(fea_num * 8, fea_num * 4)
        self.up2 = up(fea_num * 4, fea_num * 2, op='concat')
        self.up3 = up(fea_num * 2, fea_num, op='concat')
        self.outconv = outconv(fea_num, 3)

    def forward(self, fea4, skip16, skip32):
        fea8 = self.up1(fea4)
        fea16 = self.up2(fea8, skip16)
        fea32 = self.up3(fea16, skip32)
        out = self.outconv(fea32)
        return fea8, out


class Student_1(nn.Module):
    """
    student network for k=1
    """
    def __init__(self, task):
        super(Student_1, self).__init__()
        if task == 'pred':
            self.encoder = Encoder_1(12, 64)
        elif task == 'recon':
            self.encoder = Encoder_1(3, 64)
        self.decoder = Decoder_1(64)

    def forward(self, x):
        fea4, skip8, skip16, skip32 = self.encoder(x)
        out = self.decoder(fea4, skip8, skip16, skip32)
        return out, fea4


class Student_2(nn.Module):
    """
    student network for k=2
    """
    def __init__(self, task='pred'):
        super(Student_2, self).__init__()
        if task == 'pred':
            self.encoder = Encoder_2(12, 64)
        elif task == 'recon':
            self.encoder = Encoder_2(3, 64)
        self.decoder = Decoder_2(64)

    def forward(self, x):
        fea4, skip16, skip32 = self.encoder(x)
        fea8, out = self.decoder(fea4, skip16, skip32)
        return out, fea4, fea8


def weights_init_kaiming(m):
    """
    kaiming weights initialization
    """
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data, a=0.1, mode='fan_in')
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight.data, a=0.1, mode='fan_in')
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


if __name__ == '__main__':
    from torchsummary import summary

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    student = Student_2().to(device)
    summary(student, (12, 32, 32))
