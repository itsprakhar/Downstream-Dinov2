import torch
import torch.nn as nn
from torch.hub import load
import torchvision.models as models


# Define a custom classifier
class linear_head(nn.Module):
    def __init__(self, embedding_size = 384, num_classes = 5):
        super(linear_head, self).__init__()
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        return self.fc(x)


class Classifier(nn.Module):
    def __init__(self, num_classes, backbone = 'dinov2s', head = 'linear'):
        super(Classifier, self).__init__()
        self.heads = {
            'linear':linear_head
        }
        self.backbones = {
            'dinov2s':{
                'name':'dinov2_vits14',
                'embedding_size':384
            }
        }
        self.backbone = load('facebookresearch/dinov2', self.backbones[backbone]['name'])
        self.backbone.eval()
        self.head = self.heads[head](self.backbones[backbone]['embedding_size'],num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        x = self.head(x)
        return x

