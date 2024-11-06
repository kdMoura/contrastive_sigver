import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
from timm.models.vision_transformer import VisionTransformer


def linear_bn_relu(in_features, out_features):
    return nn.Sequential(OrderedDict([
        ('fc', nn.Linear(in_features, out_features, bias=False)),  # Bias is added after BN
        ('bn', nn.BatchNorm1d(out_features)),
        ('relu', nn.ReLU()),
    ]))


def build_modified_transformer(feature_space_size=2048, input_size=(150, 220)):
    base_model = VisionTransformer(img_size=input_size, in_chans=1)

    in_features=768
          
    base_model.head = nn.Sequential(OrderedDict([
            ('fc1', linear_bn_relu(in_features, 2048)),
            ('fc2', linear_bn_relu(feature_space_size, feature_space_size)),
        ]))

    base_model.feature_space_size = feature_space_size
    
    return base_model