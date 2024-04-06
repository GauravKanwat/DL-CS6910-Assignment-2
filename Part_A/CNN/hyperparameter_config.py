import argparse

def configParse():
    parser = argparse.ArgumentParser(description='Train CNN with specified parameters.')
    parser.add_argument('-wp','--wandb_project', type = str, default = 'CS6910_Assignment_2', help = 'project name')
    parser.add_argument('-we', '--wandb_entity', type = str, default='Entity', help = 'wandb entity')
    parser.add_argument('-da', '--data_augmentation', type = bool, default=True, help = 'data augmentation')
    parser.add_argument('-bs', '--batch_size', type = int, default = 32, help='batch size')
    parser.add_argument('-bn', '--batch_norm', type = bool, default = False, help='batch normalization')
    parser.add_argument('-d', '--dropout', type = int, default = 0.2, help='dropout')
    parser.add_argument('-ds', '--dense_size', type = int, default = 256, help='dense size')
    parser.add_argument('-ds', '--num_filters', type = int, default = 8, help='number of filters')
    parser.add_argument('-ds', '--filter_size', type = int, default = 5, help='filter size')
    parser.add_argument('-ds', '--activation_function', type = str, default = "GELU", help='activation function')
    parser.add_argument('-ds', '--filter_multiplier', type = float, default = 2, help='filter multiplier')
    args = parser.parse_args()

    return args
