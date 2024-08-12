import pickle
from PIL import Image

import torch
from torchvision import transforms, datasets

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import shutil

import src.infoVAE.utils as utils
from src.infoVAE.mmdVAE_train import Model



def plot_avg_loss(savepath_prefix, avg_train_losses, avg_val_losses, es_pos, y_avg):
    fig = plt.figure(figsize=(10,8))
    plt.title("Average Training and Validation Loss During Training")
    plt.plot(avg_train_losses, label='Training Loss')
    plt.plot(avg_val_losses, label='Validation Loss')

    # find position of lowest validation loss
    minposs = avg_val_losses.index(min(avg_val_losses))
    plt.axvline(minposs, linestyle='-.', color='r', label='Lowest Validation Loss Point')
    plt.axvline(es_pos, linestyle='--', color='b', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    # plt.ylim(0, 0.0012) 
    plt.ylim(0, y_avg) 
    plt.xlim(0, len(avg_train_losses)+1) 
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(savepath_prefix, 'infoVAE', 'loss-plot', 'plot_avg_loss.png'), bbox_inches='tight')


def plot_itr_loss(savepath_prefix, train_losses, val_losses, y_itr):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    ax1.plot(val_losses,label="val")
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('loss')
    ax1.set_title('Validation Loss During Training')
    ax1.set_xlim(0, len(val_losses)+1)
    # ax1.set_ylim(0, 0.002)
    ax1.set_ylim(0, y_itr)
    ax1.legend()

    ax2.plot(train_losses,label="train")
    ax2.set_xlabel('iterations')
    ax2.set_ylabel('loss')
    ax2.set_title('Training Loss During Training')
    ax2.set_xlim(0, len(train_losses)+1)
    # ax2.set_ylim(0, 0.003)
    ax2.set_ylim(0, y_itr)
    ax2.legend()

    fig.savefig(os.path.join(savepath_prefix, 'infoVAE', 'loss-plot', 'plot_itr_loss.png'))



def residual(model, savepath_prefix, num_samples, model_str, folder_path, gpu_id, use_cuda=True):
    os.makedirs(os.path.join(savepath_prefix, 'infoVAE', 'residual-plot', model_str), exist_ok=True)

    sampled_filenames = utils.sample_filename(folder_path, num_samples)

    for filename in sampled_filenames:
        # original_img = Image.open(filename).convert('RGB') # Image.open() 4 channels
        original_img = Image.open(filename)
        original_array = np.asarray(original_img) # (height, width, 3) for rgb
        # original_array = original_array / 255.0

        transform = transforms.Compose([transforms.ToTensor(),])
        processed_image = transform(original_img)
  
        test_x = processed_image.unsqueeze(0)
        test_x.requires_grad_(False)
        if(use_cuda):
            test_x = test_x.cuda(gpu_id)

        _, reconstructed_img, _, _, _ = model(test_x)

        reconstructed_array = reconstructed_img.contiguous().cpu().data.numpy()
        reconstructed_array = reconstructed_array.squeeze().transpose(1, 2, 0)
        reconstructed_array = (reconstructed_array * 255).astype(int)

        # residual_array = 1.0 - np.abs(original_array - reconstructed_array)
        # normalized = residual_array / (original_array + 1e-9)
        # normalized = (normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized))

        distance = np.abs(original_array - reconstructed_array)
        residual_array = 255 - distance

        # original_array_temp = original_array.astype(int) # to prevent wrapping around from 255 to 0
        # normalized = distance.astype(int) / (original_array_temp + 1)
        # normalized = (normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized))

        # residual_percent_array = distance / original_array
        # residual_percent_array = (residual_percent_array * 255).astype(int)
        # residual_percent_array = 255 - residual_percent_array


        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

        axes[0].imshow(original_array)
        axes[0].set_title('original image')
        axes[0].axis('off')

        axes[1].imshow(reconstructed_array)
        axes[1].set_title('reconstructed image')
        axes[1].axis('off')

        axes[2].imshow(residual_array)
        axes[2].set_title('residual image')
        axes[2].axis('off')

        # axes[3].imshow(residual_percent_array)
        # axes[3].set_title('residual image (in percentage)')
        # axes[3].axis('off')

        plt.tight_layout()

        file_names = filename.split(os.path.sep)
        savefig_path = os.path.join(savepath_prefix, 'infoVAE', 'residual-plot', model_str, f"{file_names[-1]}")
        plt.savefig(savefig_path, dpi=300)
        plt.close(fig)

