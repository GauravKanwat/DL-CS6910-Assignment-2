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
    
    layers = []         # stores the layers
    
    self.activation = getattr(nn, activation_function)()            # Activation function instance
    initial_num_filters = num_filters           # number of filters initially
    
    for i, filter_size in enumerate(filter_sizes):
        
        # First convolution layer
        if i == 0:
            num_filters = max(num_filters, 1)
            layers.append(nn.Conv2d(in_channels=3, out_channels=initial_num_filters, kernel_size=filter_size))
        
        # Second convolution layer onwards
        else:

            # Calculating the number of filters for subsequent layers
            num_filters = int(initial_num_filters * (filter_multiplier))

            # Making sure atleast one filter is used in each layer  
            num_filters = max(num_filters, 1)

            layers.append(nn.Conv2d(in_channels=initial_num_filters, out_channels=num_filters, kernel_size=filter_size))
            initial_num_filters = num_filters           # Update the initial number of filters for the next layer
            
            
        # Adding batch normalization if specified
        if batch_norm == True:
            layers.append(nn.BatchNorm2d(num_filters))
            
        layers.append(self.activation)          # Adding the activation layer
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))            # Adding maxpooling layer
    
    
    # Constructing the feature extraction module using the defined layers
    self.features = nn.Sequential(*layers)
    
    
    # Calculate the size of the feature maps after convolution and pooling
    layer_output = image_size
    for filter_size in filter_sizes:
        layer_output = (layer_output - filter_size + 1) // 2
        
    # Calculate the number of inputs which will go in first fully connected layer
    fc1_input_size = num_filters * layer_output * layer_output
    print("fc1_input: ", fc1_input_size)       
    
    
    
    # -------------Defining the values---------------

    # maxpooling layer with filter size of 2 x 2 and stride of 2
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    # Flatten layer to convert 2D feature maps into a 1D vector
    self.flatten = nn.Flatten()

    # Fully connected layer 1 with input size = fc1_input_size, and output = size dense_size
    self.fc1 = nn.Linear(fc1_input_size, dense_size)
    
    # Adding dropouts to prevent overfiting
    self.dropout = nn.Dropout(dropout)

    # Fully connected layer 2, input = dense_size, output = num_classes (which is 10 in our dataset)
    self.fc2 = nn.Linear(dense_size, num_classes)

    # -----------------------------------------------
    
  def forward(self, x):
    x = self.features(x)  # Apply convolutional and pooling layers
    x = self.flatten(x)   # Flatten the feature maps into a 1D tensor
    x = self.activation(self.fc1(x))  # Apply activation function to the first fully connected layer
    x = self.dropout(x)   # Apply dropout regularization
    x = self.fc2(x)
    # x = nn.functional.softmax(self.fc2(x), dim=1)  # Apply softmax activation to the output layer
    return x
  

def trainCNN(device, train_loader, val_loader, test_loader, model, num_epochs=10, optimizer="Adam"):    
    criterion = nn.CrossEntropyLoss()
    if optimizer == "Adam":
        opt_func = optim.Adam(model.parameters(), lr=0.001)

    


    # --------------------- Training ---------------------->

    total_correct = 0
    total_samples = 0

    for epoch in tqdm(range(num_epochs)):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)   # Inputs and labels to device

            opt_func.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass
            opt_func.step()  # Update the parameters

            # Compute the predicted labels by selecting the class index with the highest probability
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()     # Compute total correctly predicted images
            total_samples += labels.size(0)     # Total images processed

            running_loss += loss.item() * inputs.size(0)        # running loss for current epoch

        loss = running_loss / len(train_loader.dataset)     # average loss per sample (image) for the current epoch
        accuracy = total_correct / total_samples            # finding training accuracy

        print(f"Epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy * 100:.2f}%, Loss: {loss:.4f}")
        wandb.log({'accuracy': accuracy, 'loss': loss})


        # ----------------------- ** --------------------------
        




        # ------------------- Validation ---------------------->

        model.eval()    # Set the model to evaluation mode

        # Disable gradient tracking during validation
        with torch.no_grad():
            val_total_correct = 0
            val_total_samples = 0
            val_running_loss = 0.0

            for val_inputs, val_labels in tqdm(val_loader):
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)   # Inputs and labels to device

                # Perform forward pass to obtain outputs
                val_outputs = model(val_inputs)

                val_loss = criterion(val_outputs, val_labels)   # Compute validation loss

                _, val_predicted = torch.max(val_outputs, 1)    # Predictions
                val_total_correct += (val_predicted == val_labels).sum().item()     # Correctly predicted images
                val_total_samples += val_labels.size(0)         # Total images processed

                val_running_loss += val_loss.item() * val_inputs.size(0)    # validation running loss for current epoch

            val_loss = val_running_loss / len(val_loader.dataset)       # average loss per sample
            val_accuracy = val_total_correct / val_total_samples        # validation accuracy

            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {val_accuracy * 100:.2f}%, Validation Loss: {val_loss:.4f}")
            wandb.log({'val_accuracy': val_accuracy, 'val_loss': val_loss})
    
        

        # ------------------- Testing --------------------->

        # Performing testing on the last epoch, therefore after completing training and validation


        if epoch==num_epochs-1:
            model.eval()    # Set the model to evaluation mode

            with torch.no_grad():
                test_total_correct = 0
                test_total_samples = 0
                test_running_loss = 0.0
                for test_inputs, test_labels in tqdm(test_loader):
                    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                    
                    # Forward pass
                    test_outputs = model(test_inputs)

                    test_loss = criterion(test_outputs, test_labels)    # Test loss
    
                    _, test_predicted = torch.max(test_outputs, 1)      # Predictions
                    test_total_correct += (test_predicted == test_labels).sum().item()      # Correct predictions
                    test_total_samples += test_labels.size(0)           # Total images processed
    
                    test_running_loss += test_loss.item() * test_inputs.size(0)     # test running loss for current epoch
    
                test_loss = test_running_loss / len(test_loader.dataset)        # average loss per sample
                test_accuracy = test_total_correct / test_total_samples         # Test accuracy
                print(f"Test Accuracy: {test_accuracy * 100:.2f}%, Test Loss: {test_loss:.4f}")