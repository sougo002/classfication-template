from torch import nn
from torchvision import models


class EfficientNet_AdaCos(nn.Module):

    def __init__(self, cfg_model):
        super(EfficientNet_AdaCos, self).__init__()
        self.size = cfg_model.size
        self.pretrained = cfg_model.pretrained
        self.class_num = len(cfg_model.classes)

        if self.size == 'b0':
            self.backbone = models.efficientnet_b0(pretrained=self.pretrained)
        elif self.size == 'b1':
            self.backbone = models.efficientnet_b1(pretrained=self.pretrained)
        elif self.size == 'b2':
            self.backbone = models.efficientnet_b2(pretrained=self.pretrained)
        elif self.size == 'b3':
            self.backbone = models.efficientnet_b3(pretrained=self.pretrained)
        elif self.size == 'b4':
            self.backbone = models.efficientnet_b4(pretrained=self.pretrained)
        elif self.size == 'b5':
            self.backbone = models.efficientnet_b5(pretrained=self.pretrained)
        elif self.size == 'b6':
            self.backbone = models.efficientnet_b6(pretrained=self.pretrained)
        elif self.size == 'b7':
            self.backbone = models.efficientnet_b7(pretrained=self.pretrained)
        else:
            raise ValueError(f'no model size:{self.size}.')

        # 出力層をclass_numに付け替え
        self.features = nn.Sequential(
            self.backbone.features,
            self.classifier)

        # metric層をくっつける

    def foward(input, label):

# test
model = models.efficientnet_b0(pretrained=False)
print(model)