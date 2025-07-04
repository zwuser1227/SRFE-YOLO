import torch
import torch.nn as nn
import torch.nn.functional as F
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
class Decoder(nn.Module):
    def __init__(self, c1,c2):
        super(Decoder, self).__init__()
        
        self.conv1 = nn.Conv2d(c1, c1//2, 1, bias=False)
        self.conv2 = nn.Conv2d(c2, c2//2, 1, bias=False)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d((c1+c2)//2, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.ReLU(),
                                       nn.Conv2d(128, 64, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, low_level_feat,factor):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.relu(low_level_feat) 

        x = self.conv2(x)
        x = self.relu(x) 
        x = F.interpolate(x, size=[i*(factor//2) for i in low_level_feat.size()[2:]], mode='bilinear', align_corners=True)
        if factor>1:
            low_level_feat = F.interpolate(low_level_feat, size=[i*(factor//2) for i in low_level_feat.size()[2:]], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

