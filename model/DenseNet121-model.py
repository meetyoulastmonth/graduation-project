import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    """实现DenseNet中的稠密连接层"""

    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        # 稠密连接层由BN-ReLU-Conv组成
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = torch.cat([x, out], 1)  # 在通道维上进行拼接操作
        return out


class DenseBlock(nn.Module):
    """实现DenseNet中的一个稠密块"""

    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            # 构建多个稠密连接层
            layers.append(DenseLayer(in_channels + growth_rate * i, growth_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out


class TransitionLayer(nn.Module):
    """实现DenseNet中的过渡层"""

    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        # 过渡层由BN-ReLU-Conv-Pooling组成
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = self.pool(out)
        return out


class DenseNet(nn.Module):
    """实现DenseNet网络"""

    def __init__(self, num_classes=1000, growth_rate=32, block_config=(6, 12, 24, 16)):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        # 第一个卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 中间的4个稠密块
        num_features = 64
        blocks = []
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_features, growth_rate, num_layers)
            blocks.append(block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = TransitionLayer(num_features, num_features // 2)  # 缩减特征图数量
                blocks.append(trans)
                num_features = num_features // 2

        # 最后的全局平均池化、全连接层和分类器
        self.blocks = nn.Sequential(*blocks)
        self.bn_final = nn.BatchNorm2d(num_features)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        out = self.pool1(self.relu(self.bn1(self.conv1(x))))
        out = self.blocks(out)
        out = self.relu(self.bn_final(out))
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out