import numpy as np
import os
import pickle
import pandas as pd
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import src.vis.tsne as tsne
import src.vis.latent_space as latent_space
from src.infoVAE.mmdVAE import Model 



def tsne_vis(savepath_prefix, nz, model_str_dict, model_str_list):
    os.makedirs(os.path.join(savepath_prefix, 'vis', 'tsne'), exist_ok=True)

    tsne.tsne_save(savepath_prefix, nz, model_str_dict)

    tsne.plot_tsne(savepath_prefix, model_str_list)



def latent_space_vis(savepath_prefix, config, model_str_list, use_cuda=True):
    os.makedirs(os.path.join(savepath_prefix, 'vis', 'latent-space', 'range-txt'), exist_ok=True)
    for model_str in model_str_list:
        os.makedirs(os.path.join(savepath_prefix, 'vis', 'latent-space', 'dim-example', model_str), exist_ok=True)

    latent_space.check_range(savepath_prefix, config['model_params']['latent_dim'], model_str_list)

    vae = Model(config['model_params']['latent_dim'], config['model_params']['in_channels'])
    vae.load_state_dict(torch.load(os.path.join(savepath_prefix, 'infoVAE', 'checkpoint.pt')))
    if use_cuda:
        vae = vae.cuda(config['trainer_params']['gpu_id'])
    vae.eval()

    with torch.no_grad():
        for model_str in ['AGNrt', 'NOAGNrt', 'TNG100']:
            latent_space.main(savepath_prefix,
                                3.5,
                                config['model_params']['latent_dim'],
                                model_str,
                                vae,
                                config['trainer_params']['gpu_id'],
                                use_cuda=True)

