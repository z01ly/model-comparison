import torch
from torchvision import transforms, datasets
import numpy as np
import scipy
from PIL import Image
import os
import shutil
import random
import math
import pickle
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def conv_size_comp(img_size):
    F = 4 # filter size
    P = 1 # padding
    S = 2 # stride
    W = (img_size - F + 2 * P) / S + 1

    return int((W - F + 2 * P) / S + 1)
# print(conv_size_comp(28))
# print(conv_size_comp(64))


class ImageFolderWithFilenames(torch.utils.data.Dataset):
    def __init__(self, image_folder_dataset):
        self.image_folder_dataset = image_folder_dataset

    def __len__(self):
        return len(self.image_folder_dataset)

    def __getitem__(self, index):
        image, label = self.image_folder_dataset[index]
        filename = self.image_folder_dataset.imgs[index][0]
        return image, label, filename


def dataloader_func(dataroot, batch_size, workers, pin_memory, with_filename=False):
    # transforms.ToTensor(): [0, 255] to [0.0, 1.0]
    # https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html
    dataset = datasets.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                            transforms.ToTensor(),
                            ]))

    if with_filename:
        custom_dataset = ImageFolderWithFilenames(dataset)
    else:
        custom_dataset = dataset

    dataloader = torch.utils.data.DataLoader(custom_dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=workers,
                                        pin_memory=pin_memory)

    return dataloader


# early stopping class
# copy from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model, savepath_prefix):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, savepath_prefix)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            with open(os.path.join(savepath_prefix, 'infoVAE', 'output.txt'), "a") as text_file:
                text_file.write(f'EarlyStopping counter: {self.counter} out of {self.patience} \n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, savepath_prefix)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, savepath_prefix):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            with open(os.path.join(savepath_prefix, 'infoVAE', 'output.txt'), "a") as text_file:
                text_file.write(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ... \n')  
        torch.save(model.state_dict(), os.path.join(savepath_prefix, 'infoVAE', 'checkpoint.pt'))
        self.val_loss_min = val_loss


# Convert a numpy array of shape [batch_size, height, width, nc=3] into a displayable array 
# of shape [height*sqrt(batch_size), width*sqrt(batch_size)), nc=3] by tiling the images
def convert_to_display(samples):
    batch_size, height, width, nc = samples.shape
    grid_size = int(math.floor(math.sqrt(batch_size)))
    
    samples = np.reshape(samples, (grid_size, grid_size, height, width, nc))
    samples = np.transpose(samples, (0, 2, 1, 3, 4))
    samples = np.reshape(samples, (grid_size * height, grid_size * width, nc))

    return samples


def save_losses(savepath_prefix, train_losses, val_losses, avg_train_losses, avg_val_losses):
    with open(os.path.join(savepath_prefix, 'infoVAE', 'loss', 'train_losses'), "wb") as fp:
        pickle.dump(train_losses, fp)

    with open(os.path.join(savepath_prefix, 'infoVAE', 'loss', 'val_losses'), "wb") as fp:
        pickle.dump(val_losses, fp)

    with open(os.path.join(savepath_prefix, 'infoVAE', 'loss', 'avg_train_losses'), "wb") as fp:
        pickle.dump(avg_train_losses, fp)

    with open(os.path.join(savepath_prefix, 'infoVAE', 'loss', 'avg_val_losses'), "wb") as fp:
        pickle.dump(avg_val_losses, fp)


def load_losses(savepath_prefix):
    with open(os.path.join(savepath_prefix, 'infoVAE', 'loss', 'train_losses'), "rb") as fp:
        train_losses = pickle.load(fp)

    with open(os.path.join(savepath_prefix, 'infoVAE', 'loss', 'val_losses'), "rb") as fp:
        val_losses = pickle.load(fp)

    with open(os.path.join(savepath_prefix, 'infoVAE', 'loss', 'avg_train_losses'), "rb") as fp:
        avg_train_losses = pickle.load(fp)

    with open(os.path.join(savepath_prefix, 'infoVAE', 'loss', 'avg_val_losses'), "rb") as fp:
        avg_val_losses = pickle.load(fp)

    return train_losses, val_losses, avg_train_losses, avg_val_losses


def sample_filename(folder_path, num_samples):
    filenames = os.listdir(folder_path)
    sampled_filenames = random.sample(filenames, num_samples)

    join_sampled_filenames = []
    for sample in sampled_filenames:
        join_sampled_filenames.append(os.path.join(folder_path, sample))
    
    return join_sampled_filenames

