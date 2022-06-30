from torch import nn
from models import EfficientNet_AdaCos
from torchvision import models

def get_model(size: str = 'b3', class_num=2, pretrained=True):
    size = size.lower()
    model = None

    if size == 'b0':
        model = models.efficientnet_b0(pretrained=pretrained)
    elif size == 'b1':
        model = models.efficientnet_b1(pretrained=pretrained)
    elif size == 'b2':
        model = models.efficientnet_b2(pretrained=pretrained)
    elif size == 'b3':
        model = models.efficientnet_b3(pretrained=pretrained)
    elif size == 'b4':
        model = models.efficientnet_b4(pretrained=pretrained)
    elif size == 'b5':
        model = models.efficientnet_b5(pretrained=pretrained)
    elif size == 'b6':
        model = models.efficientnet_b6(pretrained=pretrained)
    elif size == 'b7':
        model = models.efficientnet_b7(pretrained=pretrained)
    else:
        raise ValueError(f'no model size:{size}.')

    # 出力層をclass_numに付け替え
    features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(features, class_num)

    return model
