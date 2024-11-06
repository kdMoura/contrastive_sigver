import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict


def linear_bn_relu(in_features, out_features):
    return nn.Sequential(OrderedDict([
        ('fc', nn.Linear(in_features, out_features, bias=False)),  # Bias is added after BN
        ('bn', nn.BatchNorm1d(out_features)),
        ('relu', nn.ReLU()),
    ]))


def build_modified_resnet(arch_name='resnet18', feature_space_size=2048):
    #base_model = models.__dict__[arch_name](pretrained=False)
    # Updated line
    base_model = models.__dict__[arch_name](weights=None)  # For no pre-trained weights
    # or if you want to use default pre-trained weights
    #base_model = models.__dict__[arch_name](weights='DEFAULT')
    
    #Modify input to single channel (grayscale)
    base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)

    if arch_name=='resnet18':
        in_features = 512
    elif arch_name=='resnet34':
        in_features = 512
    elif arch_name=='resnet50':
        in_features = 2048
    elif arch_name=='resnet101':
        in_features = 2048
    elif arch_name=='resnet152':
        in_features = 2048
          
    base_model.fc = nn.Sequential(OrderedDict([
            ('fc1', linear_bn_relu(in_features, 2048)),
            ('fc2', linear_bn_relu(feature_space_size, feature_space_size)),
        ]))

    base_model.feature_space_size = feature_space_size
    
    return base_model