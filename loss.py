import torch.nn as nn
import torch
import torch.nn.functional as F


class Generation_Loss(nn.Module):
    """
    Generation loss
    """
    def __init__(self):
        super(Generation_Loss, self).__init__()

    def forward(self, ground_truth, generation):
        return F.mse_loss(ground_truth, generation)


class Feature_Loss(nn.Module):
    """
    Feature inconsistency loss
    """
    def __init__(self):
        super(Feature_Loss, self).__init__()

    def forward(self, output, target, k):
        if k == 1:
            fea_loss = torch.mean(1 - F.cosine_similarity(output[1], target[1]))
        elif k == 2:
            fea_loss = 0.5 * torch.mean(1 - F.cosine_similarity(output[1], target[1])) + \
                       0.5 * torch.mean(1 - F.cosine_similarity(output[2], target[2]))
        return fea_loss


class Gradient_Loss(nn.Module):
    """
    Gradient loss
    """
    def __init__(self, alpha, channels, device):
        super(Gradient_Loss, self).__init__()
        self.alpha = alpha
        self.device = device
        filter = torch.FloatTensor([[-1., 1.]]).to(device)

        self.filter_x = filter.view(1, 1, 1, 2).repeat(1, channels, 1, 1)
        self.filter_y = filter.view(1, 1, 2, 1).repeat(1, channels, 1, 1)

    def forward(self, gen_frames, gt_frames):
        gen_frames_x = F.pad(gen_frames, (1, 0, 0, 0))
        gen_frames_y = F.pad(gen_frames, (0, 0, 1, 0))
        gt_frames_x = F.pad(gt_frames, (1, 0, 0, 0))
        gt_frames_y = F.pad(gt_frames, (0, 0, 1, 0))

        gen_dx = F.conv2d(gen_frames_x, self.filter_x)
        gen_dy = F.conv2d(gen_frames_y, self.filter_y)
        gt_dx = F.conv2d(gt_frames_x, self.filter_x)
        gt_dy = F.conv2d(gt_frames_y, self.filter_y)

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        return torch.mean(grad_diff_x ** self.alpha + grad_diff_y ** self.alpha)