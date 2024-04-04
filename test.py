from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pathlib
# import wandb

# wandb.login(key="0f6963d23192cbab4399ad9ec6e7475c7a0d6345")

def data_generation(dataset_path, data_augmentation=False, batch_size=32):
    
    # Mean and standard deviation values
    mean = [0.4708, 0.4596, 0.3891]
    std = [0.1951, 0.1892, 0.1859]

    # Define transformations for training and testing data
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    # Data augmentation (if data_augmentation = True) 

    augment_transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(30), 
        transforms.ToTensor(), 
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
    ])

    train_dataset = datasets.ImageFolder(root = dataset_path + "train", transform=train_transform)
    test_dataset = datasets.ImageFolder(root = dataset_path + "val", transform=test_transform)

    # Split train dataset into train and validation sets
    train_ratio = 0.8
    train_size = int(train_ratio * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_data, val_data = data_utils.random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if data_augmentation:
      augmented_dataset = datasets.ImageFolder(root = dataset_path + "train", transform=augment_transform)
      augmented_loader = torch.utils.data.DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True)
      train_loader = torch.utils.data.ConcatDataset([train_loader.dataset, augmented_loader.dataset])
      train_loader = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=True)

    # Get class names
    classpath = pathlib.Path(dataset_path + "train")
    class_names = sorted([j.name.split('/')[-1] for j in classpath.iterdir()])

    return train_loader, val_loader, test_loader, class_names

class ClassCNN(nn.Module):
  def __init__(self, num_filters, activation_function, filter_multiplier, filter_sizes, 
               dropout, batch_norm, dense_size, num_classes, image_size=256):
    super(ClassCNN, self).__init__()
        
    # Defining convolution layers
    layers = []
    params = 0
    self.activation = getattr(nn, activation_function)()
    initial_num_filters = num_filters
    
    for i, filter_size in enumerate(filter_sizes):
        
        if i == 0:
            num_filters = max(num_filters, 1)
            layers.append(nn.Conv2d(in_channels=3, out_channels=initial_num_filters, kernel_size=filter_size))
        
        else:
            num_filters = int(initial_num_filters * (filter_multiplier))     
            num_filters = max(num_filters, 1)   
            layers.append(nn.Conv2d(in_channels=initial_num_filters, out_channels=num_filters, kernel_size=filter_size))
            initial_num_filters = num_filters
            
            
        if batch_norm == True:
            layers.append(nn.BatchNorm2d(num_filters))
            
        layers.append(self.activation)
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    
    self.features = nn.Sequential(*layers)
    
    #Calculate the size of the feature maps after convolution and pooling
    final_feature_map_size = image_size
    for filter_size in filter_sizes:
        final_feature_map_size = (final_feature_map_size - filter_size + 1) // 2
        
    fc_input_size = num_filters * final_feature_map_size * final_feature_map_size
    print(fc_input_size)       
    
    # Defining maxpooling layer
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # Defining fully connected layer
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(fc_input_size, dense_size)
    self.dropout = nn.Dropout(dropout)
    self.fc2 = nn.Linear(dense_size, num_classes)
    
  def forward(self, x):
    x = self.features(x)  # Apply convolutional and pooling layers
    x = self.flatten(x)   # Flatten the feature maps into a 1D tensor
    x = self.activation(self.fc1(x))  # Apply activation function to the first fully connected layer
    x = self.dropout(x)   # Apply dropout regularization
    x = nn.functional.softmax(self.fc2(x), dim=1)  # Apply softmax activation to the output layer
    return x
  
def trainCNN(device, train_loader, val_loader, model, num_epochs=10, optimizer="Adam"):    
    criterion = nn.CrossEntropyLoss()
    if optimizer == "Adam":
        opt_func = optim.Adam(model.parameters(), lr=0.001)

    total_correct = 0
    total_samples = 0

    for epoch in tqdm(range(num_epochs)):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            opt_func.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass
            opt_func.step()  # Update the parameters

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            running_loss += loss.item() * inputs.size(0)
        loss = running_loss / len(train_loader.dataset)
        accuracy = total_correct / total_samples
        print(f"Epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy * 100:.2f}%, Loss: {loss:.4f}")
        # wandb.log({'accuracy': accuracy, 'loss': loss})

        # Validation
        model.eval()
        with torch.no_grad():
            val_total_correct = 0
            val_total_samples = 0
            val_running_loss = 0.0
            for val_inputs, val_labels in tqdm(val_loader):
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)

                _, val_predicted = torch.max(val_outputs, 1)
                val_total_correct += (val_predicted == val_labels).sum().item()
                val_total_samples += val_labels.size(0)

                val_running_loss += val_loss.item() * val_inputs.size(0)

            val_loss = val_running_loss / len(val_loader.dataset)
            val_accuracy = val_total_correct / val_total_samples
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy * 100:.2f}%, Validation Loss: {val_loss:.4f}")
            # wandb.log({'val_accuracy': val_accuracy, 'val_loss': val_loss})

def main():    
    dataset_path = './inaturalist_12K/'        
    # sweep_config = {
    #     'method' : 'bayes',                    #('random', 'grid', 'bayes')
    #     'project' : 'CS6910_Assignment_2',
    #     'metric' : {                           # Metric to optimize
    #         'name' : 'accuracy', 
    #         'goal' : 'maximize'
    #     },
    #     'parameters' : {
    #         'data_augmentation': {
    #             'values' : [True, False]
    #         },
    #         'batch_size': {
    #             'values' : [32]
    #         },
    #         'batch_norm' : {
    #             'values' : [True, False]
    #         },
    #         'dropout' : {
    #             'values' : [0.2, 0.3]
    #         },
    #         'dense_size' : {
    #             'values' : [128, 256, 512]
    #         },
    #         'num_filters' : {
    #             'values' : [4, 8, 16, 32]
    #         },
    #         'filter_size' : {
    #             'values' : [3, 5, 7]
    #         },
    #         'activation_function': {
    #             'values' : ['ReLU', 'GELU', 'SiLU', 'Mish']
    #         },
    #         'filter_multiplier': {
    #             'values' : [1, 0.5, 2]
    #         }
    #     }
    # }

    data_augmentation = False
    batch_size = 32
    batch_norm = True
    dropout = 0
    dense_size = 128
    num_filters = 4
    filter_size = 3
    activation_function = 'ReLU'
    filter_multiplier = 1
        
#     run_name = ""

    def train():   
        # with wandb.init(project="CS6910_Assignment_2") as run:
        #     config = wandb.config
        #     run_name = "aug_" + str(config.data_augmentation) + "_bs_" + str(config.batch_size) + "_norm_" + str(config.batch_norm) + "_dropout_" + str(config.dropout) + "_fc_" + str(config.dense_size) + "_nfilters_" + str(config.num_filters) +"_ac_" + config.activation_function + "_fmul_" + str(config.filter_multiplier)
        #     wandb.run.name = run_name

            train_loader, val_loader, test_loader, class_names = data_generation(dataset_path, data_augmentation, batch_size)

            print("Train: ", len(train_loader))
            print("Val: ", len(val_loader))
            print("Test: ", len(test_loader))

            # for images, labels in train_loader:
            #     show_images(images, labels)
            #     break;
            
            # filter_size = config.filter_size
            filter_sizes = []
            for i in range(5):
                filter_sizes.append(filter_size)

            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            # device = torch.device("mps" if torch.cuda.is_available() else "cpu")
            print("Device: ", device)

            print("filter size: ", filter_sizes)
            model = ClassCNN(num_filters, 
                             activation_function, 
                             filter_multiplier,
                             filter_sizes, 
                             dropout, 
                             batch_norm,
                             dense_size, 
                             num_classes=10, 
                             image_size=256)
            model.to(device)
            trainCNN(device, train_loader, val_loader, model, num_epochs=1, optimizer="Adam")
    
    # sweep_id = wandb.sweep(sweep=sweep_config)
    # wandb.agent(sweep_id, function=train, count=50)
    # wandb.finish()
    train()

main()