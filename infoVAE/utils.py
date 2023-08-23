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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sdss_size():
    filepath = "../cutouts_1000/cutouts_1000_train/3000.png"
    img = Image.open(filepath)

    print("{} x {}".format(img.height, img.width))
# 64 x 64
# sdss_size()


def conv_size_comp(img_size):
    F = 4 # filter size
    P = 1 # padding
    S = 2 # stride
    W = (img_size - F + 2 * P) / S + 1

    return int((W - F + 2 * P) / S + 1)
# print(conv_size_comp(28))
# print(conv_size_comp(64))


def NOAGN_size():
    filepath_1 = "../NOAGN/test/classic_faceon_g1.05e11.png"
    img_1 = Image.open(filepath_1)

    filepath_2 = "../NOAGN/test/classic_g1.08e11_10.png"
    img_2 = Image.open(filepath_2)

    filepath_3 = "../NOAGN/test/ell_wobh_faceon_g1.27e12.png"
    img_3 = Image.open(filepath_3)

    filepath_4 = "../NOAGN/test/ell_wobh_g1.17e13_01.png"
    img_4 = Image.open(filepath_4)


    print("{} x {} \n".format(img_1.height, img_1.width))
    print("{} x {} \n".format(img_2.height, img_2.width))
    print("{} x {} \n".format(img_3.height, img_3.width))
    print("{} x {} \n".format(img_4.height, img_4.width))
# 500 x 500 -> 64 x 64
# NOAGN_size()


def AGN_size():
    filepath_1 = "../AGN/test/bh_faceon_g1.05e11.png"
    img_1 = Image.open(filepath_1)

    filepath_2 = "../AGN/test/bh_g1.18e10_08.png"
    img_2 = Image.open(filepath_2)

    filepath_3 = "../AGN/test/ell_bh_faceon_g6.53e12.png"
    img_3 = Image.open(filepath_3)

    filepath_4 = "../AGN/test/ell_bh_g1.14e13_01.png"
    img_4 = Image.open(filepath_4)


    print("{} x {} \n".format(img_1.height, img_1.width))
    print("{} x {} \n".format(img_2.height, img_2.width))
    print("{} x {} \n".format(img_3.height, img_3.width))
    print("{} x {} \n".format(img_4.height, img_4.width))
# 500 x 500 -> 64 x 64
# AGN_size()


def n80_size():
    filepath_1 = "../n80/test/g1.37e11_n80.0_e0.13_15.png"
    img_1 = Image.open(filepath_1)

    filepath_2 = "../n80/test/g7.66e11_n80.0_e0.13_Cstar0.13_06.png"
    img_2 = Image.open(filepath_2)

    filepath_3 = "../n80/test/n80_faceon_g2.57e11_n80.0_e0.13.png"
    img_3 = Image.open(filepath_3)

    print("{} x {} \n".format(img_1.height, img_1.width))
    print("{} x {} \n".format(img_2.height, img_2.width))
    print("{} x {} \n".format(img_3.height, img_3.width))
# 500 x 500 -> 64 x 64
# n80_size()


def UHD_size():
    filepath_1 = "../UHD/test/UHD_1.12e12_06.png"
    img_1 = Image.open(filepath_1)

    filepath_2 = "../UHD/test/UHD_faceon_2.79e12.png"
    img_2 = Image.open(filepath_2)

    print("{} x {} \n".format(img_1.height, img_1.width))
    print("{} x {} \n".format(img_2.height, img_2.width))
# 500 x 500 -> 64 x 64
# UHD_size()


def downsample_mock(folder_path):
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)

        image_array = np.array(image)

        # Downsample the image to 64x64 using scipy.ndimage.zoom
        downsampled_array = scipy.ndimage.zoom(image_array, (64 / image_array.shape[0], 64 / image_array.shape[1], 1), order=3)

        downsampled_image = Image.fromarray(downsampled_array.astype(np.uint8))

        # Save the downsampled image, overwriting the original file
        downsampled_image.save(image_path)
# downsample_mock('../NOAGN/test')
# downsample_mock('../AGN/test')
# downsample_mock('../n80/test')
# downsample_mock('../UHD/test')


def sdss_split():
    source_folder = '../sdss/cutouts'
    destination_folder = '../sdss_data'

    os.makedirs(destination_folder, exist_ok=True)
    os.makedirs(os.path.join(destination_folder, "train", 'cutouts'), exist_ok=True)
    os.makedirs(os.path.join(destination_folder, "val", 'cutouts'), exist_ok=True)
    os.makedirs(os.path.join(destination_folder, "test", 'cutouts'), exist_ok=True)

    image_files = os.listdir(source_folder)
    random.shuffle(image_files)

    total_images = len(image_files)
    train_ratio = 0.7
    val_ratio = 0.2
    num_train = int(total_images * train_ratio)
    num_val = int(total_images * val_ratio)
    num_test = total_images - num_train - num_val

    train_files = image_files[: num_train]
    val_files = image_files[num_train: num_train + num_val]
    test_files = image_files[num_train + num_val: ]

    for file_ in train_files:
        shutil.copy2(os.path.join(source_folder, file_), os.path.join(destination_folder, "train", 'cutouts', file_))

    for file_ in val_files:
        shutil.copy2(os.path.join(source_folder, file_), os.path.join(destination_folder, "val", 'cutouts', file_))

    for file_ in test_files:
        shutil.copy2(os.path.join(source_folder, file_), os.path.join(destination_folder, "test", 'cutouts', file_))
# sdss_split()


def dataloader_func(dataroot, batch_size, workers, pin_memory):
    dataset = datasets.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                            transforms.ToTensor(),
                            ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=workers,
                                        pin_memory=pin_memory)

    return dataloader


# early stopping class
# copy from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='./mmdVAE_save/checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
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
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            with open("./mmdVAE_save/output.txt", "a") as text_file:
                text_file.write(f'EarlyStopping counter: {self.counter} out of {self.patience} \n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            with open("./mmdVAE_save/output.txt", "a") as text_file:
                text_file.write(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ... \n')  
        torch.save(model.state_dict(), self.path)
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


def save_losses(train_losses, val_losses, avg_train_losses, avg_val_losses):
    with open("./mmdVAE_save/loss_list/train_losses", "wb") as fp:
        pickle.dump(train_losses, fp)

    with open("./mmdVAE_save/loss_list/val_losses", "wb") as fp:
        pickle.dump(val_losses, fp)

    with open("./mmdVAE_save/loss_list/avg_train_losses", "wb") as fp:
        pickle.dump(avg_train_losses, fp)

    with open("./mmdVAE_save/loss_list/avg_val_losses", "wb") as fp:
        pickle.dump(avg_val_losses, fp)


def load_losses():
    with open("./mmdVAE_save/loss_list/train_losses", "rb") as fp:
        train_losses = pickle.load(fp)

    with open("./mmdVAE_save/loss_list/val_losses", "rb") as fp:
        val_losses = pickle.load(fp)

    with open("./mmdVAE_save/loss_list/avg_train_losses", "rb") as fp:
        avg_train_losses = pickle.load(fp)

    with open("./mmdVAE_save/loss_list/avg_val_losses", "rb") as fp:
        avg_val_losses = pickle.load(fp)

    return train_losses, val_losses, avg_train_losses, avg_val_losses


def sample_filename(folder_path):
    filenames = os.listdir(folder_path)
    num_samples = 10
    sampled_filenames = random.sample(filenames, num_samples)

    join_sampled_filenames = []
    for sample in sampled_filenames:
        join_sampled_filenames.append(os.path.join(folder_path, sample))
    
    return join_sampled_filenames
