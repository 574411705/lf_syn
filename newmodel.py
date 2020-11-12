import torch.nn as nn


class FeatureModel(nn.Module):
    def __init__(self):
        super(FeatureModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(5, 32, 3, stride=1),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ELU(),
            nn.BatchNorm2d(32)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ELU()
            nn.BatchNorm2d(32)
        )
        self.avgpool1 = nn.AvgPool2d(16, stride=1)
        self.avgpool2 = nn.AvgPool2d(8, stride=1)
        # conv5

    def forward(self, features):
        conv2 = self.layer1(features)
        conv4 = self.layer2(conv2)
        conv4 = conv2 + conv4
        pool0 = self.avgpool1(conv4)
        pool1 = self.avgpool1(conv4)
        out = nn.Concat(conv2, conv4, pool0, pool1)
        #torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)

        return out


class DisparityModel(nn.Module):
    def __init__(self):
        super(ColorNetModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(130, 128, 3, stride=1, dilation=2),
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, stride=1, dilation=4),
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, stride=1, dilation=8),
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, stride=1, dilation=16),
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, stride=1),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 4, 3, stride=1),
            nn.tanh()          
        )

    def forward(self, features):
        out = self.layer(features)
        return out * 4

class SelectionModel(nn.Module):
    def __init__(self):
        super(SelectionModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(18, 64, 3, stride=1, dilation=2),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=1, dilation=4),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(128, 128, 3, stride=1, dilation=8),
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, stride=1, dilation=16),
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, stride=1),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 4, 3, stride=1),
            nn.tanh()   
        )

    def forward(self, features):
        out = self.layer(features)
        #out = beta * out
        out = 8.01 * out
        out = nn.functional.softmax(out)
        return out
     
