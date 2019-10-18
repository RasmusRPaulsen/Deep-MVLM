import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


# 3x3 convolution with padding"
def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)


# Residual block
# Inspired from https://github.com/1adrianb/face-alignment
class ResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.resample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.resample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.resample is not None:
            residual = self.resample(residual)

        out3 += residual

        return out3


# Hour glass module
# Inspired from https://github.com/1adrianb/face-alignment
# num_features : number of output features
class HourGlassModule(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.features = num_features
        self.rb1 = ResidualBlock(self.features, self.features)
        self.rb2 = ResidualBlock(self.features, self.features)
        self.rb3 = ResidualBlock(self.features, self.features)
        self.rb4 = ResidualBlock(self.features, self.features)
        self.rb5 = ResidualBlock(self.features, self.features)
        self.rb6 = ResidualBlock(self.features, self.features)
        self.rb7 = ResidualBlock(self.features, self.features)
        self.rb8 = ResidualBlock(self.features, self.features)
        self.rb9 = ResidualBlock(self.features, self.features)
        self.rb10 = ResidualBlock(self.features, self.features)
        self.rb11 = ResidualBlock(self.features, self.features)
        self.rb12 = ResidualBlock(self.features, self.features)
        self.rb13 = ResidualBlock(self.features, self.features)
        self.rb14 = ResidualBlock(self.features, self.features)
        self.rb15 = ResidualBlock(self.features, self.features)
        self.rb16 = ResidualBlock(self.features, self.features)
        self.rb17 = ResidualBlock(self.features, self.features)
        self.rb18 = ResidualBlock(self.features, self.features)
        self.rb19 = ResidualBlock(self.features, self.features)
        self.rb20 = ResidualBlock(self.features, self.features)

    def forward(self, x):
        # example input data
        # x : (128 x 128 x 256)
        # self.features = 256

        # Upper branch
        up1 = self.rb1(x)  # up1 (128 x 128 x 256)
        lowt1 = F.max_pool2d(x, 2)  # lowt1 (64 x 64 x 256)
        low1 = self.rb2(lowt1)  # low1 (64 x 64 x 256)

        # recursion (from org code): num_down_sample = 3
        up11 = self.rb3(low1)  # up11 (64 x 64 x 256)
        lowt11 = F.max_pool2d(low1, 2)  # lowt11 (32 x 32 x 256)
        low11 = self.rb4(lowt11)  # low11 (32 x 32 x 256)

        # recursion (from org code): num_down_sample = 2
        up12 = self.rb5(low11)  # up12 (32 x 32 x 256)
        lowt12 = F.max_pool2d(low11, 2)  # lowt12 (16 x 16 x 256)
        low12 = self.rb6(lowt12)  # low12 (16 x 16 x 256)

        # recursion (from org code): num_down_sample = 1
        up13 = self.rb7(low12)  # up13 (16 x 16 x 256)
        lowt13 = F.max_pool2d(low12, 2)  # lowt13 (8 x 8 x 256)
        low13 = self.rb8(lowt13)  # low13 (8 x 8 x 256)

        # recursion (from org code): num_down_sample = 0
        up14 = self.rb9(low13)  # up14 (8 x 8 x 256)
        lowt14 = F.max_pool2d(low13, 2)  # lowt14 (4 x 4 x 256)
        low14 = self.rb10(lowt14)  # low13 (4 x 4 x 256)

        # This is the bottleneck
        low2 = self.rb11(low14)  # low2 (4 x 4 x 256)
        low3 = self.rb12(low2)  # low3 (4 x 4 x 256)
        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')  # up2 (8 x 8 x 256)
        add1 = up2 + up14  # add1 (8 x 8 x 256)

        # recursion (from org code): num_down_sample = 1
        low21 = self.rb13(add1)  # low21 (8 x 8 x 256)
        low31 = self.rb14(low21)  # low31 (8 x 8 x 256)
        up21 = F.interpolate(low31, scale_factor=2, mode='nearest')  # up2 (16 x 16 x 256)
        add2 = up21 + up13  # add2 (16 x 16 x 256)

        # recursion (from org code): num_down_sample = 2
        low22 = self.rb15(add2)  # low22 (16 x 16 x 256)
        low32 = self.rb16(low22)  # low32 (16 x 16 x 256)
        up22 = F.interpolate(low32, scale_factor=2, mode='nearest')  # up22 (32 x 32 x 256)
        add3 = up22 + up12  # add3 (32 x 32 x 256)

        # recursion (from org code): num_down_sample = 3
        low23 = self.rb17(add3)  # low23 (32 x 32 x 256)
        low33 = self.rb18(low23)  # low33 (32 x 32 x 256)
        up23 = F.interpolate(low33, scale_factor=2, mode='nearest')  # up23 (64 x 64 x 256)
        add4 = up23 + up11  # add4 (64 x 64 x 256)

        # recursion (from org code): num_down_sample = 4
        low24 = self.rb19(add4)  # low24 (64 x 64 x 256)
        low34 = self.rb20(low24)  # low34 (64 x 64 x 256)
        up24 = F.interpolate(low34, scale_factor=2, mode='nearest')  # up24 (128 x 128 x 256)
        add5 = up24 + up1  # add5 (128 x 128 x 256)

        return add5


class MVLMModel(BaseModel):
    def __init__(self, n_landmarks=73, n_features=256, dropout_rate=0.2, image_channels="geometry"):
        super().__init__()
        self.out_features = n_landmarks
        self.features = n_features
        self.dropout_rate = dropout_rate
        if image_channels == "geometry":
            self.in_channels = 1
        elif image_channels == "RGB":
            self.in_channels = 3
        elif image_channels == "depth":
            self.in_channels = 1
        elif image_channels == "RGB+depth":
            self.in_channels = 4
        elif image_channels == "geometry+depth":
            self.in_channels = 2
        else:
            print("Image channels should be: geometry, RGB, depth, RGB+depth or geometry+depth")
            self.in_channels = 1
        self.conv1 = nn.Conv2d(self.in_channels, int(self.features/4), kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(int(self.features/4))
        self.conv2 = ResidualBlock(int(self.features/4), int(self.features/2))
        self.conv3 = ResidualBlock(int(self.features/2), int(self.features/2))
        self.conv4 = ResidualBlock(int(self.features/2), self.features)
        self.hg1 = HourGlassModule(self.features)
        self.hg2 = HourGlassModule(self.features)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.conv5 = nn.Conv2d(self.features, self.features, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.features)
        self.conv6 = nn.Conv2d(self.features, self.out_features, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(self.out_features, self.features, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(self.out_features, self.out_features, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.conv9 = nn.Conv2d(self.features, self.features, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.features)
        self.conv10 = nn.Conv2d(self.features, self.out_features, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(self.out_features, self.out_features, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # assuming input images are 256 x 256 x nchannels
        # and self.features = 256
        # self.out_features = NL (number of landmarks)
        # input: x (256 x 256 x nchannels) (nchannels = 1 for geometry only)
        x = self.conv1(x)  # x: (256 x 256 x 64)
        x = self.bn1(x)
        x = F.relu(x)

        # x = F.relu(self.bn1(self.conv1(x)), True)  # x: (256 x 256 x 64)
        x = self.conv2(x)  # x: (256 x 256 x 128)
        x = F.max_pool2d(x, 2)  # x: (128 x 128 x 128)
        x = self.conv3(x)  # x: (128 x 128 x 128)
        r3 = self.conv4(x)  # r3: (128 x 128 x 256)
        x = self.hg1(r3)  # x: (128 x 128 x 256)
        x = self.dropout1(x)  # x: (128 x 128 x 256)
        ll1 = F.relu(self.bn2(self.conv5(x)), True)  # x: (128 x 128 x 256)
        x = self.conv6(ll1)  # x: (128 x 128 x NL)
        up_temp = F.interpolate(x, scale_factor=2, mode='nearest')  # up_temp (256 x 256 x NL)
        up_out = self.conv8(up_temp)  # up_out (256 x 256 x NL)
        x = self.conv7(x)  # x: (128 x 128 x 256)

        sum_temp = r3 + ll1 + x  # sum_temp: (128 x 128 x 256)

        x = self.hg2(sum_temp)
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.conv9(x)), True)  # x: (128 x 128 x 256)
        x = self.conv10(x)  # x: (128 x 128 x NL)
        up_temp2 = F.interpolate(x, scale_factor=2, mode='nearest')  # up_temp2 (256 x 256 x NL)
        up_out2 = self.conv11(up_temp2)  # up_out2 (256 x 256 x NL)

        # outputs = [up_out, up_out2]
        #        outputs.append(up_out)
        #       outputs.append(up_out2)

        outputs = torch.stack([up_out, up_out2])
        return outputs
