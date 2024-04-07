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

import hyperparameter_config
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


    # Define transformations for training, validation and testing datasets
    
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
    
    
    '''
    Split train dataset into train and validation sets
    
    '''
    
    # Create a dictionary to store the indices of samples belonging to each class in the train dataset
    train_data_class = dict()

    # Iterate over each class in the dataset
    for c in range(num_classes):
        # Get the indices of samples belonging to the current class
        train_data_class[c] = [i for i, label in enumerate(train_dataset.targets) if label == c]

    val_data_indices = []
    val_ratio = 0.2  # 20% for validation

    # Iterate over the dictionary containing indices of samples for each class
    for class_indices in train_data_class.values():
        
        # number of samples to be used for validation based on the val_ratio
        num_val = int(len(class_indices) * val_ratio)
        
        # sample 'num_val' indices from the indices belonging to the current class
        val_data_indices.extend(random.sample(class_indices, num_val))


    
    
    # Create training and validation datasets

    train_data = torch.utils.data.Subset(train_dataset, [i for i in range(len(train_dataset)) if i not in val_data_indices])
    val_data = torch.utils.data.Subset(train_dataset, val_data_indices)


    
    # Create data loaders

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
    
    ''' if data_augmentation is True, augment new dataset 
        (came from transforming training dataset) and add into 
        training dataset, therefore making the dataset considerably 
        bigger than without augmenting data
    '''
    if data_augmentation:
      augmented_dataset = datasets.ImageFolder(root = dataset_path + "train", transform=augment_transform)
      augmented_loader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True)
      train_loader = torch.utils.data.ConcatDataset([train_loader.dataset, augmented_loader.dataset])
      train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)


    # Get class names
    classpath = pathlib.Path(dataset_path + "train")
    class_names = sorted([j.name.split('/')[-1] for j in classpath.iterdir() if j.name != ".DS_Store"])

    return train_loader, val_loader, test_loader, class_names


def main(args):    
    
    # ------------- Change to your iNaturalist-12K data path -------------
    
    dataset_path = '../inaturalist_12K/' 


    
    # Sweep configuration using the bayes method and taking inputs from hyperparameter_config present in CNN folder 
    
    sweep_config = {
        'method' : 'bayes',
        'project' : args.wandb_project,
        'name' : 'Sweep',
        'entity' : args.wandb_entity,
        'metric' : {
            'name' : 'val_accuracy', 
            'goal' : 'maximize'
        },
        'parameters' : {
            'data_augmentation': {
                'values' : [args.data_augmentation]
            },
            'batch_size': {
                'values' : [args.batch_size]
            },
            'batch_norm' : {
                'values' : [args.batch_norm]
            },
            'dropout' : {
                'values' : [args.dropout]
            },
            'dense_size' : {
                'values' : [args.dense_size]
            },
            'num_filters' : {
                'values' : [args.num_filters]
            },
            'filter_size' : {
                'values' : [args.filter_size]
            },
            'activation_function': {
                'values' : [args.activation_function]
            },
            'filter_multiplier': {
                'values' : [args.filter_multiplier]
            }
        }
    }

    def train():   
        
        with wandb.init(project="CS6910_Assignment_2") as run:
            
            config = wandb.config
            run_name = "aug_" + str(config.data_augmentation) + "_bs_" + str(config.batch_size) + "_norm_" + str(config.batch_norm) + "_dropout_" + str(config.dropout) + "_fc_" + str(config.dense_size) + "_nfilters_" + str(config.num_filters) +"_ac_" + config.activation_function + "_fmul_" + str(config.filter_multiplier)
            wandb.run.name = run_name

            
            # Changing data augmentation, batch normalization according to input
            data_augmentation = True if args.data_augmentation.lower() == 'yes' else False
            batch_norm = True if args.batch_norm.lower() == 'yes' else False

            
            # Data generation and class names
            train_loader, val_loader, test_loader, class_names = data_generation(dataset_path, 
                                                                                 num_classes=10, 
                                                                                 data_augmentation=data_augmentation, 
                                                                                 batch_size=config.batch_size)
            
            
            # -------------- RUN TO VIEW TRAINING IMAGES ------------------
            
            # for images, labels in train_loader:
            #     show_images(class_names, images, labels)
            #     break

            # ------------------------- END -------------------------------
            
            
            # Creating the filter size list
            filter_sizes = []
            for i in range(5):
                filter_sizes.append(config.filter_size)
            
            
            # Torch function to switch between CPU and GPU
            
            # 2. Below code for switching between CPU and cuda
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 1. Below code for switching between CPU and Apple MPS
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            print("Device: ", device)
            
            
            # Instantiate the CNN model with the specified configuration parameters
            model = ClassCNN(num_filters=config.num_filters, 
                                activation_function=config.activation_function, 
                                filter_multiplier=config.filter_multiplier,
                                filter_sizes=filter_sizes, 
                                dropout=config.dropout, 
                                batch_norm=batch_norm,
                                dense_size=config.dense_size, 
                                num_classes=10, 
                                image_size=256)
            
            # Move the model to the specified device (e.g., cuda or mps)
            model.to(device)

            # Train the model
            trainCNN(device, train_loader, val_loader, test_loader, model, num_epochs=10, optimizer="Adam")
            
            # Show the images and the labels (pred and true)
            show_images_and_labels(device, model, test_loader, class_names)
    


    sweep_id = wandb.sweep(sweep=sweep_config)
    wandb.agent(sweep_id, function=train, count=50)
    wandb.finish()
    train()



if __name__ == "__main__":
    args = hyperparameter_config.configParse()
    main(args)