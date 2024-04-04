from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import wandb


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
    layer_output = image_size
    for filter_size in filter_sizes:
        layer_output = (layer_output - filter_size + 1) // 2
        
    fc1_input_size = num_filters * layer_output * layer_output
    print("fc1_input: ", fc1_input_size)       
    
    # Defining maxpooling layer
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    # Defining fully connected layer
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(fc1_input_size, dense_size)
    self.dropout = nn.Dropout(dropout)
    self.fc2 = nn.Linear(dense_size, num_classes)
    
  def forward(self, x):
    x = self.features(x)  # Apply convolutional and pooling layers
    x = self.flatten(x)   # Flatten the feature maps into a 1D tensor
    x = self.activation(self.fc1(x))  # Apply activation function to the first fully connected layer
    x = self.dropout(x)   # Apply dropout regularization
    x = self.fc2(x)
    # x = nn.functional.softmax(self.fc2(x), dim=1)  # Apply softmax activation to the output layer
    return x
  

def trainCNN(device, train_loader, val_loader, test_loader, model, testing_mode=True, num_epochs=10, optimizer="Adam"):    
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
            return model


def testCNN(device, test_loader, model):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        test_loss = 0.0
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            test_loss += loss.item() * inputs.size(0)
        loss = test_loss / len(test_loader.dataset)
        accuracy = total_correct / total_samples
        return accuracy, loss, model