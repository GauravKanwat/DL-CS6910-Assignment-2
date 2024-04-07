from tqdm.auto import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import wandb

wandb.login()

import sys
sys.path.append('../Part_A/CNN')

from ClassCNN import trainCNN

def show_images(class_names, images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        plt.subplot(6, 6, i + 1)

        # Transpose the image tensor to (height, width, channels) for displaying
        plt.imshow(np.transpose(images[i], (1, 2, 0)))
        plt.title(f"{class_names[labels[i]]}")
        plt.axis('off')
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    plt.show()



def data_generation(dataset_path, num_classes=10, data_augmentation=False, batch_size=32):
    
    # Mean and standard deviation values calculated from function get_mean_and_std

    mean = [0.4708, 0.4596, 0.3891]
    std = [0.1951, 0.1892, 0.1859]


    # Define transformations for training and testing data
    
    augment_transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(30), 
        transforms.ToTensor()
        # transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
        # transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
        # transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])


    # Data augmentation (if data_augmentation = True) 

    train_dataset = datasets.ImageFolder(root = dataset_path + "train", transform=train_transform)
    test_dataset = datasets.ImageFolder(root = dataset_path + "val", transform=test_transform)
    
    
    # Split train dataset into train and validation sets

    train_data_class = dict()
    for c in range(num_classes):
        train_data_class[c] = [i for i, label in enumerate(train_dataset.targets) if label == c]

    val_data_indices = []
    val_ratio = 0.2  # 20% for validation
    for class_indices in train_data_class.values():
        num_val = int(len(class_indices) * val_ratio)
        val_data_indices.extend(random.sample(class_indices, num_val))


    # Create training and validation datasets

    train_data = torch.utils.data.Subset(train_dataset, [i for i in range(len(train_dataset)) if i not in val_data_indices])
    val_data = torch.utils.data.Subset(train_dataset, val_data_indices)


    # Create data loaders

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if data_augmentation:
      augmented_dataset = datasets.ImageFolder(root = dataset_path + "train", transform=augment_transform)
      augmented_loader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True)
      train_loader = torch.utils.data.ConcatDataset([train_loader.dataset, augmented_loader.dataset])
      train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)


    # Get class names
    classpath = pathlib.Path(dataset_path + "train")
    class_names = sorted([j.name.split('/')[-1] for j in classpath.iterdir() if j.name != ".DS_Store"])

    return train_loader, val_loader, test_loader, class_names



# ----------Performe fine-tuning on the pre-trained model----------->


def feature_extraction(model, device):
    
    #Freeze all the layers
    for params in model.parameters():
        params.requires_grad = False

def freeze_till_k(model, device, k):
    # Counter to track the number of frozen layers
    frozen_layers = 0
    
    for param in model.parameters():
        # Freeze layers up to the k-th layer
        if frozen_layers < k:
            param.requires_grad = False
            frozen_layers += 1
        else:
            # Stop freezing layers after k-th layer
            break


def no_freezing(model, device):

    # Unfreeze all the layers
    for params in model.parameters():
        params.requires_grad = True


#----------------------END------------------------------------------>




def main():    
    dataset_path = '../inaturalist_12K/' 

    data_augmentation = True
    batch_size = 32
    num_classes = 10
    fine_tuning_method = 2      # Fine-tuning method
    k = 12                      # Number of layers to freeze

    def train():   
        with wandb.init(project="CS6910_Assignment_2_Part_B") as run:
            config = wandb.config
            run_name = "aug_" + str(data_augmentation) + "_bs_" + str(batch_size) + "_fine_tune_" + str(fine_tuning_method) + "_num_freeze_layer_all"
            if fine_tuning_method != 1:
                run_name = "aug_" + str(data_augmentation) + "_bs_" + str(batch_size) + "_fine_tune_" + str(fine_tuning_method) + "_num_freeze_layer_" + str(k)
            elif fine_tuning_method == 3:
                run_name = "aug_" + str(data_augmentation) + "_bs_" + str(batch_size) + "_fine_tune_" + str(fine_tuning_method) + "_num_freeze_layer_none"

            wandb.run.name = run_name
            train_loader, val_loader, test_loader, class_names = data_generation(dataset_path, 
                                                                                     num_classes=10, 
                                                                                     data_augmentation=data_augmentation, 
                                                                                     batch_size=batch_size)
            print("Train: ", len(train_loader))
            print("Val: ", len(val_loader))
            print("Test: ", len(test_loader))


            # -------------- RUN TO VIEW TRAINING IMAGES ------------------
            
            # for images, labels in train_loader:
            #     show_images(class_names, images, labels)
            #     break

            # ------------------------- END -------------------------------

            
            filter_sizes = []
            for i in range(5):
                filter_sizes.append(config.filter_size)
            
            # Torch function to switch between CPU and GPU
            
            # 2. Below code for switching between CPU and cuda
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 1. Below code for switching between CPU and Apple MPS
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            print("Device: ", device)

            
            model = models.googlenet(pretrained=True)
            model.to(device)

            # Method 1: Feature Extraction
            if fine_tuning_method == 1:
                feature_extraction(model, device)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                model.to(device)
                trainCNN(device, train_loader, val_loader, test_loader, model, num_epochs=5, optimizer="Adam")
            
            # Method 2: Freeze till k layers
            elif fine_tuning_method == 2:
                freeze_till_k(model, device, k)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                model.to(device)
                trainCNN(device, train_loader, val_loader, test_loader, model, num_epochs=5, optimizer="Adam")

            # Method 3: No freezing
            else:
                feature_extraction(model, device)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                model.to(device)
                trainCNN(device, train_loader, val_loader, test_loader, model, num_epochs=5, optimizer="Adam")
    
    wandb.finish()
    train()

main()
