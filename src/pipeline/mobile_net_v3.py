import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, squeeze_channels):
        super(SqueezeExcitation, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.hsigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.hsigmoid(scale)
        return x * scale


class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 expand_ratio, se_ratio=0.25, activation='RE'):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = stride == 1 and in_channels == out_channels

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_se = se_ratio is not None and se_ratio > 0

        layers = []

        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish() if activation == 'HS' else nn.ReLU(inplace=True)
            ])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                      padding=(kernel_size - 1) // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Hardswish() if activation == 'HS' else nn.ReLU(inplace=True)
        ])

        if self.use_se:
            squeeze_channels = max(1, int(in_channels * se_ratio))
            layers.append(SqueezeExcitation(hidden_dim, squeeze_channels))

        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, num_classes=7, width_multiplier=1.0):
        super(MobileNetV3, self).__init__()

        input_channels = int(16 * width_multiplier)
        self.stem = nn.Sequential(
            nn.Conv2d(3, input_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.Hardswish(inplace=True)
        )

        bneck_configs = [
            [16, 16, 3, 1, 1, None, 'RE'],
            [16, 24, 3, 2, 4, None, 'RE'],
            [24, 24, 3, 1, 3, None, 'RE'],
            [24, 40, 5, 2, 3, 0.25, 'RE'],
            [40, 40, 5, 1, 3, 0.25, 'RE'],
            [40, 40, 5, 1, 3, 0.25, 'RE'],
            [40, 80, 3, 2, 6, None, 'HS'],
            [80, 80, 3, 1, 2.5, None, 'HS'],
            [80, 80, 3, 1, 2.3, None, 'HS'],
            [80, 80, 3, 1, 2.3, None, 'HS'],
            [80, 112, 3, 1, 6, 0.25, 'HS'],
            [112, 112, 3, 1, 6, 0.25, 'HS'],
            [112, 160, 5, 2, 6, 0.25, 'HS'],
            [160, 160, 5, 1, 6, 0.25, 'HS'],
            [160, 160, 5, 1, 6, 0.25, 'HS'],
        ]

        layers = []
        for config in bneck_configs:
            in_ch = int(config[0] * width_multiplier)
            out_ch = int(config[1] * width_multiplier)
            layers.append(InvertedResidual(
                in_ch, out_ch, config[2], config[3],
                config[4], config[5], config[6]
            ))
        self.bottlenecks = nn.Sequential(*layers)

        last_channels = int(160 * width_multiplier)
        self.head_conv = nn.Sequential(
            nn.Conv2d(last_channels, 960, 1, bias=False),
            nn.BatchNorm2d(960),
            nn.Hardswish(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.bottlenecks(x)
        x = self.head_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


def MobileNetV3_Large(num_classes=7):
    return MobileNetV3(num_classes=num_classes, width_multiplier=1.0)
