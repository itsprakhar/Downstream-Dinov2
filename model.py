import torch
import torch.nn as nn
from torch.hub import load
import torchvision.models as models

# Load pre-trained DINO model
dino_model = load('facebookresearch/dinov2', 'dinov2_vits14')
dino_model.eval()

# Define a custom classifier
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.dino_model = dino_model
        self.fc = nn.Linear(384, num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.dino_model(x)
        x = self.fc(x)
        return x
