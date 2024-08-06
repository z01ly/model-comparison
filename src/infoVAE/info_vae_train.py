# ref: https://github.com/Bjarten/early-stopping-pytorch

import torch

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import yaml

import src.infoVAE.utils as utils
from src.infoVAE.info_vae import InfoVAE


def train(savepath_prefix, train_dataloader, val_dataloader):
    with open('src/infoVAE/infovae.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model = InfoVAE(config['model_params']['in_channels'],
                    config['model_params']['latent_dim'],
                    None,
                    config['model_params']['alpha'],
                    config['model_params']['beta'],
                    config['model_params']['reg_weight'],
                    config['model_params']['kernel_type'],
                    config['model_params']['latent_var'])

    model = model.cuda(config['trainer_params']['gpu_id'])

    optimizer = torch.optim.Adam(model.parameters())
    scheduler_patience = int(config['trainer_params']['patience'] / 2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=scheduler_patience)

    train_losses = []
    val_losses = []
    avg_train_losses = [] # average training loss per epoch
    avg_val_losses = [] # average validation loss per epoch
    early_stopping = utils.EarlyStopping(patience=config['trainer_params']['patience'], verbose=True, delta=0.000008)

    n_epochs = config['trainer_params']['max_epochs']
    for epoch in range(n_epochs):
        # train the model
        model.train()
        sum_train_loss_epoch = 0
        itr = 0
        for batch_idx, (images, _) in enumerate(train_dataloader):
            x = images
            x.requires_grad_(False)
            x = x.cuda(config['trainer_params']['gpu_id'])
            
            loss_list = model(x)
            loss_dict = {'M_N': config['exp_params']['kld_weight'], 'current_device': config['trainer_params']['gpu_id']}
            loss = model.loss_function(*loss_list, **loss_dict)

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
                x = images
                x.requires_grad_(False)
                x = x.cuda(config['trainer_params']['gpu_id'])
                
                loss_list = model(x)
                loss_dict = {'M_N': config['exp_params']['kld_weight'], 'current_device': config['trainer_params']['gpu_id']}
                loss = model.loss_function(*loss_list, **loss_dict)

                val_losses.append(loss.item())
                sum_val_loss_epoch += loss.item()
                itr += 1

            # average val loss per epoch
            val_avg = sum_val_loss_epoch / itr
            avg_val_losses.append(val_avg)

        # adaptive learning rate
        scheduler.step(val_avg)

        # sample 64 images in each epoch
        samples = model.sample(64, config['trainer_params']['gpu_id'])
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
            
    return train_losses, val_losses, avg_train_losses, avg_val_losses

