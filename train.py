# Importing necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
from tqdm import tqdm
# Import the Classifier class from model.py
from model import Classifier

# Define data augmentation and normalization for training
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Randomly resize and crop the images to 224x224
    transforms.RandomHorizontalFlip(),  # Randomly flip the images horizontally
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the tensors with mean and standard deviation
])

# Load data
data_dir = 'data/train'
full_dataset = datasets.ImageFolder(data_dir, transform)  # Load images and labels from the directory
train_size = int(0.8 * len(full_dataset))  # 80% of the dataset for training
val_size = len(full_dataset) - train_size  # 20% for validation

# Split the data into training and validation datasets
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create data loaders for training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

dataloaders = {'train': train_loader, 'val': val_loader}

# Main execution
if __name__ == "__main__":

    # Set the device to GPU if available, otherwise CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = len(full_dataset.classes)
    # Initialize the model and send it to the device
    model = Classifier(num_classes).to(device)

    # Define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=7, verbose=True)

    # Define early stopping parameters
    n_epochs_stop = 6
    epochs_no_improve = 0
    min_val_loss = np.Inf

    # Training loop
    for epoch in range(100):
        print('Epoch {}/{}'.format(epoch, 100 - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            correct = 0
            total = 0

            # Progress bar setup with tqdm
            with tqdm(total=len(dataloaders[phase]), unit='batch') as p:
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Backward pass and optimize only in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # Gather training statistics - loss and accuracy
                    running_loss += loss.item() * inputs.size(0)
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()

                    # Update tqdm progress bar
                    p.set_postfix({'loss': loss.item(), 'accuracy': 100 * correct / total})
                    p.update(1)

            # Compute the average loss and accuracy over the entire epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = 100 * correct / total

            print('{} Loss: {:.4f} Acc: {:.2f}%'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            # If the model's validation loss is lower than our current minimum,
            # save this model as the best model so far
            if phase == 'val':
                if epoch_loss < min_val_loss:
                    print(f'Validation Loss Decreased({min_val_loss:.6f}--->{epoch_loss:.6f}) \t Saving The Model')
                    min_val_loss = epoch_loss
                    # Save the model
                    torch.save(model.state_dict(), 'weights/saved_model.pt')
                    epochs_no_improve = 0
                else:
                    # If the validation loss didn't improve, increase the stop counter
                    epochs_no_improve += 1
                    # If the validation loss hasn't improved in several epochs, stop the training early
                    if epochs_no_improve == n_epochs_stop:
                        print('Early stopping!')
                        # Load the last best model saved
                        model.load_state_dict(torch.load('weights/saved_model.pt')) 
                        model.eval()
                        break  # exit the loop
        # Step the learning rate scheduler
        scheduler.step(epoch_loss)  # update learning rate scheduler

