import torchvision.models as models
import torch.nn as nn
import torch
import os

model_dirs = {
    'resnet': './teacher/resnet50-0676ba61.pth',
    'resnext': './teacher/resnext50_32x4d-7cdf4587.pth',
    'wideresnet': './teacher/wide_resnet50_2-95faca4d.pth',
}

model_urls = {
    'resnet': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnext': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'wideresnet': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
}


def download_resnet(backbone_name='resnet'):
    """
    download teacher backbone
    """
    if not os.path.exists(model_dirs[backbone_name]):
        print('Downloading teacher network')
        torch.hub.download_url_to_file(model_urls[backbone_name], model_dirs[backbone_name])


class Teacher(nn.Module):
    """
    teacher network
    output teacher features
    """
    def __init__(self, backbone_name='resnet'):
        super(Teacher, self).__init__()
        self.backbone_name = backbone_name
        self.block1, self.block2, self.block3 = self._get_backbone()

    def _get_backbone(self):
        if self.backbone_name == 'resnet':
            backbone = models.resnet50(pretrained=False)
            backbone.load_state_dict(torch.load(model_dirs[self.backbone_name]))
        elif self.backbone_name == 'resnext':
            backbone = models.resnext50_32x4d(pretrained=False)
            backbone.load_state_dict(torch.load(model_dirs[self.backbone_name]))
        elif self.backbone_name == 'wideresnet':
            backbone = models.wide_resnet50_2(pretrained=False)
            backbone.load_state_dict(torch.load(model_dirs[self.backbone_name]))
        else:
            raise NotImplementedError('Use resnet, resnext or wildresnet as teacher network')

        block1 = nn.Sequential(*list(backbone.children())[:-5])
        block2 = nn.Sequential(*list(backbone.children())[-5:-4])
        block3 = nn.Sequential(*list(backbone.children())[-4:-3])

        return block1, block2, block3

    def forward(self, x):
        fea8 = self.block1(x)
        fea4 = self.block2(fea8)
        fea2 = self.block3(fea4)

        return fea2, fea4, fea8


if __name__ == '__main__':
    from torchsummary import summary
    backbone_name = 'wideresnet'
    download_resnet(backbone_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    teacher = Teacher(backbone_name).to(device)
    summary(teacher, (3, 32, 32))

