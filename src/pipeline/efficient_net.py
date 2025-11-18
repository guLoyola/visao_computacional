import torch
import torch.nn as nn
import math
from collections import namedtuple


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_channels):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1),
            Swish(),
            nn.Conv2d(reduced_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 expand_ratio, se_ratio=0.25, drop_connect_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = (stride == 1 and in_channels == out_channels)

        hidden_dim = in_channels * expand_ratio
        self.expand = expand_ratio != 1

        if self.expand:
            self.expansion = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish()
            )

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                      padding=kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Swish()
        )

        reduced_channels = max(1, int(in_channels * se_ratio))
        self.se = SqueezeExcitation(hidden_dim, reduced_channels)

        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x

        if self.expand:
            x = self.expansion(x)

        x = self.depthwise(x)
        x = self.se(x)

        x = self.project(x)

        if self.use_residual:
            if self.training and self.drop_connect_rate > 0:
                x = self._drop_connect(x, self.drop_connect_rate)
            x = x + identity

        return x

    def _drop_connect(self, x, drop_rate):
        if not self.training:
            return x
        keep_prob = 1 - drop_rate
        batch_size = x.shape[0]
        random_tensor = keep_prob
        random_tensor += torch.rand([batch_size, 1, 1, 1],
                                    dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x / keep_prob * binary_tensor


BlockConfig = namedtuple('BlockConfig', [
                         'expand_ratio', 'channels', 'num_blocks', 'stride', 'kernel_size'])

EFFICIENTNET_CONFIGS = {
    'b0': [
        BlockConfig(1, 16, 1, 1, 3),
        BlockConfig(6, 24, 2, 2, 3),
        BlockConfig(6, 40, 2, 2, 5),
        BlockConfig(6, 80, 3, 2, 3),
        BlockConfig(6, 112, 3, 1, 5),
        BlockConfig(6, 192, 4, 2, 5),
        BlockConfig(6, 320, 1, 1, 3),
    ],
    'b1': [
        BlockConfig(1, 16, 1, 1, 3),
        BlockConfig(6, 24, 2, 2, 3),
        BlockConfig(6, 40, 2, 2, 5),
        BlockConfig(6, 80, 3, 2, 3),
        BlockConfig(6, 112, 3, 1, 5),
        BlockConfig(6, 192, 4, 2, 5),
        BlockConfig(6, 320, 1, 1, 3),
    ],
    'b2': [
        BlockConfig(1, 16, 1, 1, 3),
        BlockConfig(6, 24, 2, 2, 3),
        BlockConfig(6, 40, 2, 2, 5),
        BlockConfig(6, 80, 3, 2, 3),
        BlockConfig(6, 112, 3, 1, 5),
        BlockConfig(6, 192, 4, 2, 5),
        BlockConfig(6, 320, 1, 1, 3),
    ],
    'b3': [
        BlockConfig(1, 16, 1, 1, 3),
        BlockConfig(6, 24, 2, 2, 3),
        BlockConfig(6, 40, 2, 2, 5),
        BlockConfig(6, 80, 3, 2, 3),
        BlockConfig(6, 112, 3, 1, 5),
        BlockConfig(6, 192, 4, 2, 5),
        BlockConfig(6, 320, 1, 1, 3),
    ],
}

EFFICIENTNET_PARAMS = {
    'b0': {'width_mult': 1.0, 'depth_mult': 1.0, 'resolution': 224, 'dropout': 0.2},
    'b1': {'width_mult': 1.0, 'depth_mult': 1.1, 'resolution': 240, 'dropout': 0.2},
    'b2': {'width_mult': 1.1, 'depth_mult': 1.2, 'resolution': 260, 'dropout': 0.3},
    'b3': {'width_mult': 1.2, 'depth_mult': 1.4, 'resolution': 300, 'dropout': 0.3},
}


class EfficientNet(nn.Module):
    def __init__(self, version='b0', num_classes=7, drop_connect_rate=0.2):
        super(EfficientNet, self).__init__()

        if version not in EFFICIENTNET_CONFIGS:
            raise ValueError(
                f"Version {version} not supported. Choose from: {list(EFFICIENTNET_CONFIGS.keys())}")

        params = EFFICIENTNET_PARAMS[version]
        width_mult = params['width_mult']
        depth_mult = params['depth_mult']
        dropout_rate = params['dropout']

        out_channels = self._round_filters(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            Swish()
        )

        block_configs = EFFICIENTNET_CONFIGS[version]
        in_channels = out_channels
        blocks = []
        total_blocks = sum(config.num_blocks for config in block_configs)
        block_idx = 0

        for config in block_configs:
            num_blocks = self._round_repeats(config.num_blocks, depth_mult)
            out_channels = self._round_filters(config.channels, width_mult)

            for i in range(num_blocks):
                drop_rate = drop_connect_rate * block_idx / total_blocks

                blocks.append(MBConvBlock(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=config.kernel_size,
                    stride=config.stride if i == 0 else 1,
                    expand_ratio=config.expand_ratio,
                    se_ratio=0.25,
                    drop_connect_rate=drop_rate
                ))
                block_idx += 1

            in_channels = out_channels

        self.blocks = nn.Sequential(*blocks)

        head_channels = self._round_filters(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 1, bias=False),
            nn.BatchNorm2d(head_channels),
            Swish()
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(head_channels, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def _round_filters(self, filters, width_mult):
        if width_mult == 1.0:
            return filters
        filters *= width_mult
        divisor = 8
        new_filters = max(divisor, int(
            filters + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def _round_repeats(self, repeats, depth_mult):
        if depth_mult == 1.0:
            return repeats
        return int(math.ceil(depth_mult * repeats))

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


def EfficientNet_B0(num_classes=7):
    return EfficientNet(version='b0', num_classes=num_classes, drop_connect_rate=0.2)


def EfficientNet_B1(num_classes=7):
    return EfficientNet(version='b1', num_classes=num_classes, drop_connect_rate=0.2)


def EfficientNet_B2(num_classes=7):
    return EfficientNet(version='b2', num_classes=num_classes, drop_connect_rate=0.2)


def EfficientNet_B3(num_classes=7):
    return EfficientNet(version='b3', num_classes=num_classes, drop_connect_rate=0.2)
