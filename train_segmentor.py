from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
from model import Segmentor
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from tools.segmentation import SegmentationDataset, train, validation, infer


img_transform = transforms.Compose([
    transforms.Resize((14*32,14*32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mask_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
])



dataset = SegmentationDataset(img_dir=r"data\segmentation\train\imgs", mask_dir=r"data\segmentation\train\labels", num_classes = 5, img_transform=img_transform, mask_transform=mask_transform)


# Splitting data into train and validation sets
train_imgs, valid_imgs = train_test_split(dataset.images, test_size=0.2, random_state=42)

train_dataset = SegmentationDataset(img_dir=r"data\segmentation/train/imgs", mask_dir=r"data\segmentation/train/labels", num_classes = 5, img_transform=img_transform, mask_transform=mask_transform, images=train_imgs)
valid_dataset = SegmentationDataset(img_dir=r"data\segmentation/train/imgs", mask_dir=r"data\segmentation/train/labels", num_classes = 5, img_transform=img_transform, mask_transform=mask_transform, images=valid_imgs)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True) #, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=1) #, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Segmentor(5)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()


num_epochs = 1
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, epoch)
    validation(model, criterion, valid_loader)


torch.save(model.state_dict(), 'weights/segmentation_model.pt')

