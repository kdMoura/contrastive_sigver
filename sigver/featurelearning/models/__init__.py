from .signet import SigNet, SigNet_smaller, SigNet_thin
from torchvision.models.resnet import ResNet
from timm.models.vision_transformer import VisionTransformer

available_models = {'signet': SigNet,
                    'signet_thin': SigNet_thin,
                    'signet_smaller': SigNet_smaller,
                    'resnet18': ResNet,
                    'resnet34': ResNet,
                    'resnet50': ResNet,
                    'resnet101': ResNet,
                    'resnet152': ResNet,
                    'vit': VisionTransformer}
