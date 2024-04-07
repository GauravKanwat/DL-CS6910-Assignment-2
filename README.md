# CS6910 Assignment 2

## Part A

- `Train_PartA.py`: Containing main function for training the convolutional neural network
  
- `Part_A/CNN/`
  
  - `Requirements.txt`: txt file containing all the Python libraries needed to train CNN.
    
  - `hyperparameter_config.py`: the file containing all the hyperparameters and their default values.
    
  - `ClassCNN.py`: contains the class `ClassCNN` and all the functions that define the CNN architectures used.
 
- `Part_A/`
  - `CNN.ipynb`: ipynb file of `Train_PartA.py`

<br>

### Instructions for running the Convolutional Neural Network code
To train the CNN, please follow the steps given below:

- Import the required libraries:
   ```
   pip install -r Part_A/CNN/Requirements.txt

- Please put your Wandb API key in `Train_PartA.py` before running the file to track the runs.

   
- Run the below code to run on default parameters (check the end of file for default parameters).
   ```
   python Part_A/Train_PartA.py
   
- Use your parameters:
    - Example: `Python Part_A/Train_PartA.py --activation_function SiLU` to run the CNN with activation function `SiLU`.

<br>

Link to the wandb report: [Link](https://wandb.ai/cs23m024-gaurav/CS6910_Assignment_2/reports/Copy-of-oikantik-s-CS6910-Assignment-2--Vmlldzo3NDA0MDY5)

<br>

### Dataset

- I have used the `iNaturalist-12K` dataset. It consists of 12,000 RGB images of various species from the natural world. These images cover a wide range of categories, including animals, plants, insects, and more. The images in this dataset have variable sizes and resolutions, reflecting the diversity of photographic sources. Each image is labeled with its corresponding species to facilitate classification tasks.
- To prepare the iNaturalist-12K dataset for training, we first split the data into training and validation sets. This was done to ensure that the model learns from a diverse range of images while also being able to generalize well to unseen data.
- After splitting the data, we performed data normalization and transformation. Normalization involved scaling the pixel values of the images to a range that is suitable for training neural networks, typically between 0 and 1.
- Transformation included resizing the images to a uniform size and converting them into tensors, which are the primary data structure used in PyTorch.
- Additionally, for some experimental runs, we applied data augmentation techniques to the training set. Augmentation involved applying random transformations such as rotations, flips, and shifts to the training images. This helped to increase the variability of the training data and improve the model's ability to generalize to new examples.

<br>

## Part B

- `Train_PartB.py`: Containing main function for training the convolutional neural network
  
- `Part_B/`
  
  - `googLeNet.ipynb`: file containing the explanation of Part B.


### Instructions for running the Convolutional Neural Network on pre-trained model
To fine-tune the CNN, please follow the steps given below:


- Please put your Wandb API key in `Train_PartB.py` before running the file to track the runs.

   
- Run the below code to run on default parameters (check the end of file for default parameters).
   ```
   python Part_B/Train_PartB.py

### Hyperparameters and their default values for Part A
| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | CS6910_Assignment_1 | Project name used to track experiments in Weights & Biases dashboard |
| `-we`,  `--wandb_entity` | CS23M024  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-da`, `--data_augmentation` | Yes | Whether to augment data or not |
| `-bs`, `--batch_size` | 32 | Batch size to train the dataset. |
| `-bn`, `--batch_norm` | No | Perform batch normalization after every convolution layer |
| `-d`, `--dropout` | 0.2 | choices:  [0, 0.2, 0.3] | 
| `-ds`, `--dense_size` | 256 | choices: [128, 256, 512] | 
| `-nf`, `--num_filters` | 0.9 | choices: [4, 8, 16, 32] |
| `-fs`, `--filter_size` | 0.9 | choices: [3, 5, 7] | 
| `-af`, `--activation_function` | 0.9 | choices: ["ReLU", "GELU", "SiLU","Mish"] | 
| `-fm`, `--filter_multiplier` | 0.999 | choices: [0, 0.2, 0.3] |
<br>
