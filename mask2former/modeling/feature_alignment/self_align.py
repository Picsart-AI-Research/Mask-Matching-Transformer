# MySelfAlignLayer : self alignment block in MM-Former
import torch, os, cv2, math
from torch import nn
from torch.nn import functional as F

class MySelfAlignLayer(nn.Module):
    def __init__(self, minsize = 15):
        super().__init__()
        self.minsize = minsize
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, que, sup=None):
        if sup is None:
            return self.spatial_pool(que)
        return self.spatial_pool(que), self.spatial_pool(sup)

    def channel_pool(self, x):
        # [N, IC, H, W]
        g_x = x
        batch, channel, height, width = g_x.size()

        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)
        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, H*W]
        theta_x = x.view(batch, channel, height * width)
        theta_x = self.softmax(theta_x)

        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = torch.matmul(avg_x, theta_x)

        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)
        print(mask_sp.shape)
        z,_ = torch.max(mask_sp, dim = 1)
        print(z)

        out = x * mask_sp + x

        return out

    def spatial_pool(self, x):
        input_x = x

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = torch.mean(input_x, dim = 1)
        context_mask = context_mask.unsqueeze(1)

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax(context_mask)

        # [N, IC, 1]
        context = torch.matmul(input_x, context_mask.transpose(1, 2))

        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)
        out = x * mask_ch

        return out