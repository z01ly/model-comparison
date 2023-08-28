import utils
import pickle
from mmdVAE_train import Model
from PIL import Image

import torch
from torch.autograd import Variable

from torchvision import transforms, datasets

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import shutil
from sklearn.manifold import TSNE



def plot_avg_loss(avg_train_losses, avg_val_losses, pos):
    fig = plt.figure(figsize=(10,8))
    plt.title("Average Training and Validation Loss During Training")
    plt.plot(avg_train_losses, label='Training Loss')
    plt.plot(avg_val_losses,label='Validation Loss')

    # find position of lowest validation loss
    minposs = avg_val_losses.index(min(avg_val_losses))
    plt.axvline(minposs, linestyle='--', color='r',label='Lowest Validation Loss Point')
    plt.axvline(pos, linestyle='--', color='b',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.0012) 
    plt.xlim(0, len(avg_train_losses)+1) 
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig('./mmdVAE_save/plot_avg_loss.png', bbox_inches='tight')


def plot_itr_loss(train_losses, val_losses):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    ax1.plot(val_losses,label="val")
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('loss')
    ax1.set_title('Validation Loss During Training')
    ax1.set_xlim(0, len(val_losses)+1)
    ax1.set_ylim(0, 0.002)
    ax1.legend()

    ax2.plot(train_losses,label="train")
    ax2.set_xlabel('iterations')
    ax2.set_ylabel('loss')
    ax2.set_title('Training Loss During Training')
    ax2.set_xlim(0, len(train_losses)+1)
    ax2.set_ylim(0, 0.003)
    ax2.legend()

    fig.savefig('./mmdVAE_save/plot_itr_loss.png')



def plot_residual(model, folder_path, gpu_id, use_cuda=True, sdss=False):
    if not sdss:
        last_num = -6
    else:
        last_num = -14

    try:
        shutil.rmtree(os.path.join('./test_results/residual/', folder_path[3: last_num]))
    except:
        pass
    os.makedirs(os.path.join('./test_results/residual/', folder_path[3:])) # e.g. './test_results/residual/NOAGN/test/'

    sampled_filenames = utils.sample_filename(folder_path) # e.g. '../NOAGN/test/classic_g3.67e10_18.png'

    for filename in sampled_filenames:
        original_img = Image.open(filename).convert('RGB') # Image.open() 4 channels
        original_array = np.asarray(original_img) # (height, width, 3) for rgb
        # original_array = original_array / 255.0

        transform = transforms.Compose([transforms.ToTensor(),])
        processed_image = transform(original_img)
  
        test_x = Variable(processed_image.unsqueeze(0), requires_grad=False)
        if(use_cuda):
            test_x = test_x.cuda(gpu_id)

        z, reconstructed_img = model(test_x)

        reconstructed_array = reconstructed_img.contiguous().cpu().data.numpy()
        reconstructed_array = reconstructed_array.squeeze().transpose(1, 2, 0)
        reconstructed_array = (reconstructed_array * 255).astype(int)

        # residual_array = 1.0 - np.abs(original_array - reconstructed_array)
        # normalized = residual_array / (original_array + 1e-9)
        # normalized = (normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized))

        distance = np.abs(original_array - reconstructed_array)
        residual_array = 255 - distance

        original_array_temp = original_array.astype(int) # to prevent wrapping around from 255 to 0
        normalized = distance.astype(int) / (original_array_temp + 1)
        normalized = (normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized))


        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 5))

        axes[0].imshow(original_array)
        axes[0].set_title('original image')
        axes[0].axis('off')

        axes[1].imshow(reconstructed_array)
        axes[1].set_title('reconstructed image')
        axes[1].axis('off')

        axes[2].imshow(residual_array)
        axes[2].set_title('residual image')
        axes[2].axis('off')

        axes[3].imshow(normalized)
        axes[3].set_title('normalized image')
        axes[3].axis('off')

        plt.tight_layout()

        savefig_path = os.path.join('./test_results/residual/', filename[3:])
        plt.savefig(savefig_path, dpi=300)

        plt.close(fig)


def plot_tsne(dim=2, include_sdss=False):
    fig = plt.figure(figsize=(10,10))
    if dim == 3:
        ax = fig.add_subplot(111, projection='3d')

    colors = ['b', 'r', 'g', 'k', 'y']
    markers = ['o', 'v', 's', '*', '+']
    i = 0
    for filename_latent in os.listdir('./test_results/latent/'):
        if (not include_sdss) and (filename_latent == 'sdss_test.npy'):
            continue

        if (filename_latent == 'UHD_10times.npy') or (filename_latent == 'n80_5times.npy'):
            continue

        latent_z = np.load('./test_results/latent/' + filename_latent) # e.g. shape of UHD: (126, 32)
        z_embedded = TSNE(n_components=dim, perplexity=100, init='pca', random_state=42).fit_transform(latent_z)

        if dim == 2:
            plt.scatter(z_embedded[:, 0], z_embedded[:, 1], s=5, c=colors[i], marker=markers[i], label=filename_latent[: -4])
        elif dim == 3:
            ax.scatter(z_embedded[:, 0], z_embedded[:, 1], z_embedded[:, 2], s=5, c=colors[i], \
            marker=markers[i], label=filename_latent[: -4])
        i += 1

    if dim == 2:
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title("Applying tsne to latent z")
        plt.legend()
    elif dim == 3:
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('3D Scatter Plot')
        ax.legend()
    
    if not include_sdss:
        fig.savefig('./test_results/plot_tsne_models.png')
    else:
        fig.savefig('./test_results/plot_tsne_all.png')




if __name__ == '__main__':
    gpu_id = 2
    image_size = 64
    nc = 3
    nz = 32
    z_dim = nz
    n_filters = 64
    after_conv = utils.conv_size_comp(image_size)
    use_cuda = True

    model = Model(z_dim, nc, n_filters, after_conv)
    model.load_state_dict(torch.load('./mmdVAE_save/checkpoint.pt'))
    if use_cuda:
        model = model.cuda(gpu_id)
    model.eval()


    # residual plots
    with torch.no_grad():
        folder_paths = ['../NOAGN/test/', '../AGN/test/', '../n80/test/', '../UHD/test/']
        for folder_path in folder_paths:
            plot_residual(model, folder_path, gpu_id, use_cuda, False)
        plot_residual(model, '../sdss_data/test/cutouts/', gpu_id, use_cuda, True)
    

    # training losses
    train_losses, val_losses, avg_train_losses, avg_val_losses = utils.load_losses()
    plot_avg_loss(avg_train_losses, avg_val_losses, 22)
    plot_itr_loss(train_losses, val_losses)
    
    
    # tsne
    plot_tsne(2)
    plot_tsne(2, True)





