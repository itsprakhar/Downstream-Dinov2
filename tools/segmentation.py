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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, num_classes, img_transform=None, mask_transform=None, images=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.num_classes = num_classes

        # Only include images for which a mask is found
        if images is None:
            self.images = [img for img in os.listdir(img_dir) if os.path.isfile(os.path.join(mask_dir, img.split(".")[0] + ".png"))]
        else:
            self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx].split(".")[0]
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, img_name + ".png")
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask) * 255
        # Create a tensor to hold the binary masks
        bin_mask = torch.zeros(self.num_classes, mask.shape[1], mask.shape[2])

        # Ensure mask is a torch tensor and is in the same device as bin_mask
        mask = torch.from_numpy(np.array(mask)).to(bin_mask.device)
        
        # Convert mask to type float for comparison
        mask = mask.float()

        for i in range(self.num_classes):
            bin_mask[i] = (mask == i).float()  # Ensure resulting mask is float type

        return image, bin_mask



def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    loop = tqdm(train_loader, total=len(train_loader))
    running_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(loop):
        # print(batch_idx) 
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss = loss.item())

    print(f'\nTrain set: Average loss: {running_loss/len(train_loader):.4f}')


def validation(model, criterion, valid_loader):
    model.eval()
    running_loss = 0
    correct = 0

    with torch.no_grad():
        loop = tqdm(valid_loader, total=len(valid_loader))
        for data, target in loop:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)

    print(f'\nValidation set: Average loss: {running_loss/len(valid_loader):.4f}')


def infer(image_path, model, device, img_transform):
    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    transformed_image = img_transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Make sure the model is in evaluation mode
    model.eval()

    with torch.no_grad():
        # Make prediction
        output = model(transformed_image)

        # Get the predicted class for each pixel
        _, predicted = torch.max(output, 1)
    
    # Move prediction to cpu and convert to numpy array
    predicted = predicted.squeeze().cpu().numpy()

    return transformed_image.cpu().squeeze().permute(1, 2, 0).numpy(), predicted
