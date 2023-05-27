# Import necessary modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from model import Classifier  # Import custom model from model.py file

# Define a function for loading and transforming image data
def load_data(data_dir='data/train'):
    # Define transformations: random crop, random flip, convert to tensor, and normalize
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # Resize and crop the image to a 224x224 square
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the image with mean and standard deviation
    ])

    # Load the dataset from directory and apply transformations
    full_dataset = datasets.ImageFolder(data_dir, transform)

    # Split the full dataset into train and validation sets
    train_size = int(0.8 * len(full_dataset))  # 80% of the dataset is used for training
    val_size = len(full_dataset) - train_size  # The rest is used for validation

    # Randomly split dataset into training and validation dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create data loaders for train and validation sets
    # They provide an easy way to iterate over the dataset in mini-batches
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Shuffle the training data
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # No need to shuffle validation data

    return {'train': train_loader, 'val': val_loader}, len(full_dataset.classes)  # Return loaders and number of classes in the dataset

# Define a function to train the model
def train_model(model, dataloaders, criterion, optimizer, scheduler, device, n_epochs_stop=6):
    # Initialize variables
    min_val_loss = np.Inf  # Minimum validation loss starts at infinity
    epochs_no_improve = 0  # No improvement in epochs counter

    # Loop over epochs
    for epoch in range(100):
        print('Epoch {}/{}'.format(epoch, 100 - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Initialize metrics for this phase
            running_loss = 0.0  # Accumulate losses over the epoch
            correct = 0  # Count correct predictions
            total = 0  # Count total predictions

            # Use tqdm for progress bar
            with tqdm(total=len(dataloaders[phase]), unit='batch') as p:
                # Iterate over mini-batches
                for inputs, labels in dataloaders[phase]:
                    # Move input and label tensors to the default device (GPU or CPU)
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Clear the gradients of all optimized variables
                    optimizer.zero_grad()

                    # Forward pass: compute predicted outputs by passing inputs to the model
                    with torch.set_grad_enabled(phase == 'train'):  # Only calculate gradients in training phase
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)  # Get the class with the highest probability
                        loss = criterion(outputs, labels)  # Compute the loss

                        # Perform backward pass and optimization only in the training phase
                        if phase == 'train':
                            loss.backward()  # Calculate gradients based on the loss
                            optimizer.step()  # Update model parameters based on the current gradient

                    # Update running loss and correct prediction count
                    running_loss += loss.item() * inputs.size(0)  # Multiply average loss by batch size
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()  # Update correct predictions count

                    # Update the progress bar
                    p.set_postfix({'loss': loss.item(), 'accuracy': 100 * correct / total})
                    p.update(1)

                # Calculate loss and accuracy for this epoch
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = 100 * correct / total

                print('{} Loss: {:.4f} Acc: {:.2f}%'.format(phase, epoch_loss, epoch_acc))

                # Save model and implement early stopping in validation phase
                if phase == 'val':
                    if epoch_loss < min_val_loss:
                        print(f'Validation Loss Decreased({min_val_loss:.6f}--->{epoch_loss:.6f}) \t Saving The Model')
                        min_val_loss = epoch_loss  # Update minimum validation loss
                        torch.save(model.state_dict(), 'weights/saved_model.pt')  # Save the current model weights
                        epochs_no_improve = 0  # Reset epochs since last improvement
                    else:
                        epochs_no_improve += 1
                        # Implement early stopping
                        if epochs_no_improve == n_epochs_stop:
                            print('Early stopping!')
                            model.load_state_dict(torch.load('weights/saved_model.pt'))  # Load the best model weights
                            model.eval()
                            return model  # Exit the function early

            # Adjust the learning rate based on the scheduler
            scheduler.step(epoch_loss)

    # Return the trained model
    return model


if __name__ == "__main__":
    # Decide whether to use a GPU or not
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    dataloaders, num_classes = load_data()

    # Initialize model, loss function (criterion), optimizer, and learning rate scheduler
    model = Classifier(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=7, verbose=True)

    # Train the model
    model = train_model(model, dataloaders, criterion, optimizer, scheduler, device)
