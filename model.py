# Importing the necessary libraries
import torch
import torch.nn as nn
from torch.hub import load
import torchvision.models as models

# Creating a class 'linear_head' that extends the nn.Module class
# This class will act as the head of our model, meaning it's the final piece of the model which will produce predictions
class linear_head(nn.Module):
    def __init__(self, embedding_size = 384, num_classes = 5):
        super(linear_head, self).__init__()
        # This line is defining a fully connected (linear) layer which takes 'embedding_size' inputs and outputs 'num_classes' values
        self.fc = nn.Linear(embedding_size, num_classes)

    # This is the forward method, defining the forward pass of the linear_head
    def forward(self, x):
        return self.fc(x)

# Creating a class 'Classifier' that extends the nn.Module class
# This will be our final classification model
class Classifier(nn.Module):
    def __init__(self, num_classes, backbone = 'dinov2_s', head = 'linear'):
        super(Classifier, self).__init__()
        # Defining the types of heads our model can have. Here, we have only one type: 'linear'
        self.heads = {
            'linear':linear_head
        }
        # Defining the backbones our model can have with their corresponding names and embedding sizes
        self.backbones = {
            'dinov2_s':{
                'name':'dinov2_vits14',
                'embedding_size':384
            },
            'dinov2_b':{
                'name':'dinov2_vitb14',
                'embedding_size':768
            },
            'dinov2_l':{
                'name':'dinov2_vitl14',
                'embedding_size':1024
            },
            'dinov2_g':{
                'name':'dinov2_vitg14',
                'embedding_size':1536
            },
        }
        # Loading the backbone model using torch.hub.load
        self.backbone = load('facebookresearch/dinov2', self.backbones[backbone]['name'])
        # Setting the backbone to evaluation mode
        self.backbone.eval()
        # Initializing the head of the model
        self.head = self.heads[head](self.backbones[backbone]['embedding_size'],num_classes)

    # This is the forward method, defining the forward pass of the Classifier
    def forward(self, x):
        # The input first goes through the backbone. We use 'torch.no_grad()' to avoid calculating gradients during the forward pass of the backbone
        with torch.no_grad():
            x = self.backbone(x)
        # Then it goes through the head
        x = self.head(x)
        return x
