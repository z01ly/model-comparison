# ref: https://github.com/Bjarten/early-stopping-pytorch

import torch

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import os
import yaml

import src.infoVAE.utils as utils
from src.infoVAE.mmdVAE import Model


# Training
def train(savepath_prefix, train_dataloader, val_dataloader, use_cuda=True):
    with open('src/infoVAE/infovae.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model = Model(config)
    if use_cuda:
        model = model.cuda(config['trainer_params']['gpu_id'])

    optimizer = torch.optim.Adam(model.parameters())
    scheduler_patience = int(config['trainer_params']['patience'] / 2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=scheduler_patience)

    train_losses = []
    val_losses = []
    avg_train_losses = [] # average training loss per epoch
    avg_val_losses = [] # average validation loss per epoch
    early_stopping = utils.EarlyStopping(patience=config['trainer_params']['patience'], verbose=True, delta=0.000005)

    for epoch in range(config["trainer_params"]["max_epochs"]):
        if epoch < 7:
            k_pre_value = config['model_params']['k_pre_value'] + config['model_params']["k_step"] * (7 - epoch)
        else:
            k_pre_value = config['model_params']['k_pre_value']
        print(f"k_pre_value in epoch {epoch}: {k_pre_value}")

        # train the model
        model.train()
        sum_train_loss_epoch = 0
        itr = 0
        for batch_idx, (images, _) in enumerate(train_dataloader):
            x = images
            x.requires_grad_(False)
            true_samples = torch.randn(len(images), config['model_params']['latent_dim'])
            true_samples.requires_grad_(False)
            if use_cuda:
                x = x.cuda(config['trainer_params']['gpu_id'])
                true_samples = true_samples.cuda(config['trainer_params']['gpu_id'])
            
            z_sparse, x_reconstructed, mu, log_var, _ = model(x, k_pre_value)

            loss_dict = model.loss_func(true_samples, z_sparse, x_reconstructed, x, mu, log_var)
            loss = loss_dict['loss']

            optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
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
                x = images
                x.requires_grad_(False)
                true_samples = torch.randn(len(images), config['model_params']['latent_dim'])
                true_samples.requires_grad_(False)
                if use_cuda:
                    x = x.cuda(config['trainer_params']['gpu_id'])
                    true_samples = true_samples.cuda(config['trainer_params']['gpu_id'])
                
                z_sparse, x_reconstructed, mu, log_var, _ = model(x, k_pre_value)

                loss_dict = model.loss_func(true_samples, z_sparse, x_reconstructed, x, mu, log_var)
                loss = loss_dict['loss']

                val_losses.append(loss.item())
                sum_val_loss_epoch += loss.item()
                itr += 1

            # average val loss per epoch
            val_avg = sum_val_loss_epoch / itr
            avg_val_losses.append(val_avg)

        # adaptive learning rate
        scheduler.step(val_avg)

        # sample 64 images in each epoch
        gen_z = torch.randn(64, config['model_params']['latent_dim'])
        gen_z.requires_grad_(False)
        if use_cuda:
            gen_z = gen_z.cuda(config['trainer_params']['gpu_id'])
        samples = model.decoder(gen_z)
        samples = samples.permute(0,2,3,1).contiguous().cpu().data.numpy()
        plt.imshow(utils.convert_to_display(samples)) # imshow for rgb images
        savefig_path = os.path.join(savepath_prefix, 'infoVAE', 'samples', 'fig_epoch' + str(epoch) + '.png')
        plt.savefig(savefig_path, dpi=300)

        # print message during training
        epoch_len = len(str(config["trainer_params"]["max_epochs"]))
        print_msg = (f'[{epoch:>{epoch_len}}/{config["trainer_params"]["max_epochs"]:>{epoch_len}}] ' +
                    f'train_avg: {train_avg:.5f} ' +
                    f'val_avg: {val_avg:.5f}')
        print(print_msg)
        with open(os.path.join(savepath_prefix, 'infoVAE', 'output.txt'), "a") as text_file:
            text_file.write((f'[{epoch:>{epoch_len}}/{config["trainer_params"]["max_epochs"]:>{epoch_len}}] ' +
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

