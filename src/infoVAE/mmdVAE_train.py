# ref: https://github.com/Bjarten/early-stopping-pytorch

import torch
from torch.autograd import Variable
from torchvision import transforms, datasets

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import os 

import src.infoVAE.utils as utils
from src.infoVAE.mmdVAE import Model


# mmd loss function
def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd


# Training
def train(savepath_prefix, train_dataloader, val_dataloader, patience, 
        z_dim=2, nc=3, n_filters=64, after_conv=16, n_epochs=10, use_cuda=True, gpu_id=0):
    model = Model(z_dim, nc, n_filters, after_conv)
    if use_cuda:
        model = model.cuda(gpu_id)

    optimizer = torch.optim.Adam(model.parameters())
    scheduler_patience = int(patience / 2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=scheduler_patience)

    train_losses = []
    val_losses = []
    avg_train_losses = [] # average training loss per epoch
    avg_val_losses = [] # average validation loss per epoch
    early_stopping = utils.EarlyStopping(patience=patience, verbose=True, delta=0.000008)

    for epoch in range(n_epochs):
        # train the model
        model.train()
        sum_train_loss_epoch = 0
        itr = 0
        for batch_idx, (images, _) in enumerate(train_dataloader):
            # x = Variable(images, requires_grad=False)
            x = images
            x.requires_grad_(False)
            # true_samples = Variable(torch.randn(len(images), z_dim), requires_grad=False) # len(images) samples
            true_samples = torch.randn(len(images), z_dim)
            true_samples.requires_grad_(False)
            if use_cuda:
                x = x.cuda(gpu_id)
                true_samples = true_samples.cuda(gpu_id)
            
            z, x_reconstructed = model(x)
            mmd = compute_mmd(true_samples, z)
            nll = (x_reconstructed - x).pow(2).mean()
            loss = nll + mmd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            sum_train_loss_epoch += loss.item()
            itr += 1
        
        # average train loss per epoch
        train_avg = sum_train_loss_epoch / itr
        avg_train_losses.append(train_avg)
        
        # validate the model
        model.eval()
        sum_val_loss_epoch = 0
        itr = 0
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(val_dataloader):
                # x = Variable(images, requires_grad=False)
                x = images
                x.requires_grad_(False)
                # true_samples = Variable(torch.randn(len(images), z_dim), requires_grad=False)
                true_samples = torch.randn(len(images), z_dim)
                true_samples.requires_grad_(False)
                if use_cuda:
                    x = x.cuda(gpu_id)
                    true_samples = true_samples.cuda(gpu_id)
                
                z, x_reconstructed = model(x)
                mmd = compute_mmd(true_samples, z)
                nll = (x_reconstructed - x).pow(2).mean()
                loss = nll + mmd

                val_losses.append(loss.item())
                sum_val_loss_epoch += loss.item()
                itr += 1

            # average val loss per epoch
            val_avg = sum_val_loss_epoch / itr
            avg_val_losses.append(val_avg)

        # adaptive learning rate
        scheduler.step(val_avg)

        # sample 64 images in each epoch
        # gen_z = Variable(torch.randn(64, z_dim), requires_grad=False)
        gen_z = torch.randn(64, z_dim)
        gen_z.requires_grad_(False)
        if use_cuda:
            gen_z = gen_z.cuda(gpu_id)
        samples = model.decoder(gen_z)
        samples = samples.permute(0,2,3,1).contiguous().cpu().data.numpy()
        plt.imshow(utils.convert_to_display(samples)) # imshow for rgb images
        savefig_path = os.path.join(savepath_prefix, 'infoVAE', 'samples', 'fig_epoch' + str(epoch) + '.png')
        plt.savefig(savefig_path, dpi=300)

        # print message during training
        epoch_len = len(str(n_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                    f'train_avg: {train_avg:.5f} ' +
                    f'val_avg: {val_avg:.5f}')
        print(print_msg)
        with open(os.path.join(savepath_prefix, 'infoVAE', 'output.txt'), "a") as text_file:
            text_file.write((f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                    f'train_avg: {train_avg:.5f} ' +
                    f'val_avg: {val_avg:.5f} \n'))

        # early stopping
        early_stopping(val_avg, model, savepath_prefix)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint
    # model.load_state_dict(torch.load('src/infoVAE/mmdVAE_save/checkpoint.pt'))

    # return model, train_losses, val_losses, avg_train_losses, avg_val_losses
    return train_losses, val_losses, avg_train_losses, avg_val_losses



# Main
if __name__ == "__main__":
    gpu_id = 1
    workers = 4
    batch_size = 400
    image_size = 64 # sdss image size
    nc = 3 # number of input and output channels. 3 for color images.
    nz = 32 # Size of z latent vector
    n_filters = 64 # Size of feature maps
    num_epochs = 100 # Number of training epochs
    after_conv = utils.conv_size_comp(image_size)
    patience = 10

    train_dataroot = 'src/data/sdss_data/train'
    train_dataloader = utils.dataloader_func(train_dataroot, batch_size, workers, False)

    val_dataroot = 'src/data/sdss_data/val'
    val_dataloader = utils.dataloader_func(val_dataroot, batch_size, workers, False)
    
    model, train_losses, val_losses, avg_train_losses, avg_val_losses = \
            train(train_dataloader=train_dataloader, val_dataloader=val_dataloader, patience=patience, 
                z_dim=nz, nc=nc, n_filters=n_filters, after_conv=after_conv, n_epochs=num_epochs, 
                use_cuda=True, gpu_id=gpu_id)
    
    utils.save_losses(train_losses, val_losses, avg_train_losses, avg_val_losses)
