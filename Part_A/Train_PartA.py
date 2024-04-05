from tqdm.auto import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import wandb

wandb.login()

import sys
sys.path.append('CNN')

from ClassCNN import ClassCNN, trainCNN

def get_mean_and_std(train_loader):
    mean = 0
    std = 0
    total_images_count = 0
    for images, _ in train_loader:
        images_count_in_a_batch = images.size(0)
        images = images.view(images_count_in_a_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += images_count_in_a_batch
    mean /= total_images_count
    std /= total_images_count
    return mean, std


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

def show_images_and_labels(device, model, test_loader, class_names):
    model.eval()
    with torch.no_grad():  # Disable gradient tracking
        images_per_class = {class_name: 0 for class_name in class_names}
        fig, axes = plt.subplots(10, 3, figsize=(15, 30))  # 10x3 grid
        
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for image, label, pred in zip(images, labels, predicted):
                class_name = class_names[label.item()]
                if images_per_class[class_name] < 3:
                    ax = axes[label.item(), images_per_class[class_name]]
                    img = image.permute(1, 2, 0).cpu().numpy()
                    ax.imshow(img)
                    ax.set_title(f"Predicted: {class_names[pred.item()]}\nOriginal: {class_name}")
                    ax.axis('off')
                    images_per_class[class_name] += 1
            
            if all(count == 3 for count in images_per_class.values()):
                break
                
        # Prevent overlap
        plt.tight_layout()
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
        transforms.ToTensor(), 
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
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


def main():    
    dataset_path = '../inaturalist_12K/' 

    sweep_config = {
        'method' : 'bayes',                    #('random', 'grid', 'bayes')
        'project' : 'CS6910_Assignment_2',
        'metric' : {                           # Metric to optimize
            'name' : 'accuracy', 
            'goal' : 'maximize'
        },
        'parameters' : {
            'data_augmentation': {
                'values' : [True, False]
            },
            'batch_size': {
                'values' : [32]
            },
            'batch_norm' : {
                'values' : [True, False]
            },
            'dropout' : {
                'values' : [0.2, 0.3]
            },
            'dense_size' : {
                'values' : [128, 256, 512]
            },
            'num_filters' : {
                'values' : [4, 8, 16, 32]
            },
            'filter_size' : {
                'values' : [3, 5, 7]
            },
            'activation_function': {
                'values' : ['ReLU', 'GELU', 'SiLU', 'Mish']
            },
            'filter_multiplier': {
                'values' : [1, 0.5, 2]
            }
        }
    }

    def train():   
        with wandb.init(project="CS6910_Assignment_2") as run:
            config = wandb.config
            run_name = "aug_" + str(config.data_augmentation) + "_bs_" + str(config.batch_size) + "_norm_" + str(config.batch_norm) + "_dropout_" + str(config.dropout) + "_fc_" + str(config.dense_size) + "_nfilters_" + str(config.num_filters) +"_ac_" + config.activation_function + "_fmul_" + str(config.filter_multiplier)
            wandb.run.name = run_name

            train_loader, val_loader, test_loader, class_names = data_generation(dataset_path, 
                                                                                 num_classes=10, 
                                                                                 data_augmentation=config.data_augmentation, 
                                                                                 batch_size=config.batch_size)
            
            # for images, labels in train_loader:
            #     show_images(class_names, images, labels)
            #     break
            
            filter_sizes = []
            for i in range(5):
                filter_sizes.append(config.filter_size)
            
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            print("Device: ", device)
            
            print("filter size: ", filter_sizes)
            model = ClassCNN(num_filters=config.num_filters, 
                                activation_function=config.activation_function, 
                                filter_multiplier=config.filter_multiplier,
                                filter_sizes=filter_sizes, 
                                dropout=config.dropout, 
                                batch_norm=config.batch_norm,
                                dense_size=config.dense_size, 
                                num_classes=10, 
                                image_size=256)
            model.to(device)

            trainCNN(device, train_loader, val_loader, test_loader, model, num_epochs=10, optimizer="Adam")
            
            show_images_and_labels(device, model, test_loader, class_names)
    
    sweep_id = wandb.sweep(sweep=sweep_config)
    wandb.agent(sweep_id, function=train, count=50)
    wandb.finish()
    train()

main()