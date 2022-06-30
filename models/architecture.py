import torch
import torch.nn.functional as F
from torchvision import models

import math

class EfficientNet(torch.nn.Module):
    def __init__(self, size: str = 'b3', out_features=2, pretrained=False):
        super(EfficientNet, self).__init__()
        size = size.lower()
        if size == 'b0':
            self.model = models.efficientnet_b0(pretrained=pretrained)
        elif size == 'b1':
            self.model = models.efficientnet_b1(pretrained=pretrained)
        elif size == 'b2':
            self.model = models.efficientnet_b2(pretrained=pretrained)
        elif size == 'b3':
            self.model = models.efficientnet_b3(pretrained=pretrained)
        elif size == 'b4':
            self.model = models.efficientnet_b4(pretrained=pretrained)
        elif size == 'b5':
            self.model = models.efficientnet_b5(pretrained=pretrained)
        elif size == 'b6':
            self.model = models.efficientnet_b6(pretrained=pretrained)
        elif size == 'b7':
            self.model = models.efficientnet_b7(pretrained=pretrained)
        else:
            raise ValueError(f'no model size:{size}.')

        # 出力層をout_featuresに付け替え
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = torch.nn.Linear(in_features, out_features)

    def freeze_bn(self):
        for m in self.features.modules():
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def forward(self, input):
        output = self.model(input)
        return output


class EfficientNet_AdaCos(torch.nn.Module):

    def __init__(self, cfg_model):
        super(EfficientNet_AdaCos, self).__init__()
        self.cfg_model = cfg_model
        self.size = cfg_model.size
        self.pretrained = cfg_model.pretrained
        self.class_num = len(cfg_model.classes) if len(cfg_model.classes) > 2 else 3
        self.feature_size = cfg_model.feature_size

        self.backbone = EfficientNet(self.size, self.feature_size, self.pretrained)
        # 特徴抽出の部分
        self.features = self.backbone.model.features
        self.last_channels = self.backbone.model.classifier[-1].in_features
        # 出力層付け替え
        self.bn1 = torch.nn.BatchNorm2d(self.last_channels)
        self.dropout = torch.nn.Dropout2d(0.5)
        self.fc = torch.nn.Linear(self.last_channels*((self.cfg_model.image_size//32)**2), self.feature_size)

        self.bn2 = torch.nn.BatchNorm1d(self.feature_size)

        # metric層のパラメタ初期化
        self.s = math.sqrt(2) * math.log(self.class_num - 1)
        self.m = cfg_model.params.m
        self.W = torch.nn.Parameter(torch.FloatTensor(self.class_num, self.feature_size))
        torch.nn.init.xavier_uniform_(self.W)


    def forward(self, input, label=None):
        # output features(base model)
        x = self.features(input)
        x = self.bn1(x)
        x = self.dropout(x)
        # bsize, 1536, image/32 ,image/32 -> bsize, -1
        x = x.view(input.shape[0], -1)
        x = self.fc(x)
        features = self.bn2(x)

        # normalize features
        x = F.normalize(features)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # このへん理解してないので誰か教えて下さい
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits

        return output
