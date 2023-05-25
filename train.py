import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
from tqdm import tqdm
from model import Classifier

# Dataloader
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = 'data/train'
full_dataset = datasets.ImageFolder(data_dir, transform)
train_size = int(0.8 * len(full_dataset))  # 80% for training
val_size = len(full_dataset) - train_size  # 20% for validation

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) #, num_workers=4, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) #, num_workers=4, persistent_workers=True)

dataloaders = {'train': train_loader, 'val': val_loader}

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = len(full_dataset.classes)
    model = Classifier(num_classes).to(device)

    # Define loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=7, verbose=True)

    # Early stopping parameters
    n_epochs_stop = 6
    epochs_no_improve = 0
    min_val_loss = np.Inf

    # Training loop
    for epoch in range(100):  # for the sake of example, we train for 100 epochs
        print('Epoch {}/{}'.format(epoch, 100 - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            correct = 0
            total = 0

            # Iterate over data with tqdm progress bar
            with tqdm(total=len(dataloaders[phase]), unit='batch') as p:
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()

                    # Update tqdm progress bar
                    p.set_postfix({'loss': loss.item(), 'accuracy': 100 * correct / total})
                    p.update(1)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = 100 * correct / total

            print('{} Loss: {:.4f} Acc: {:.2f}%'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                if epoch_loss < min_val_loss:
                    print(f'Validation Loss Decreased({min_val_loss:.6f}--->{epoch_loss:.6f}) \t Saving The Model')
                    min_val_loss = epoch_loss
                    torch.save(model.state_dict(), 'weights/saved_model.pt')
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    # Check early stopping condition
                    if epochs_no_improve == n_epochs_stop:
                        print('Early stopping!')
                        model.load_state_dict(torch.load('weights/saved_model.pt'))  # load the last best model
                        model.eval()
                        break  # exit the loop
        scheduler.step(epoch_loss)  # update learning rate scheduler
