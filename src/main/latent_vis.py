import numpy as np
from PIL import Image
import os
import random
import math
import pickle
import pandas as pd
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import src.vis.tsne as tsne
import src.vis.latent_space as latent_space
from src.infoVAE.mmdVAE import Model 
from src.infoVAE.utils import conv_size_comp



def tsne_vis(savepath_prefix, nz, model_str_dict, model_str_list):
    os.makedirs(os.path.join(savepath_prefix, 'vis', 'tsne'), exist_ok=True)
    tsne.tsne_save(savepath_prefix, nz, model_str_dict)

    tsne.sn_plot_tsne(savepath_prefix, model_str_list)



def latent_space_vis(savepath_prefix, gpu_id, nz, model_str_list, use_cuda=True):
    os.makedirs(os.path.join(savepath_prefix, 'vis', 'latent-space', 'range-txt'), exist_ok=True)
    for model_str in model_str_list:
        os.makedirs(os.path.join(savepath_prefix, 'vis', 'latent-space', 'dim-example', model_str), exist_ok=True)

    latent_space.check_range(savepath_prefix, nz, model_str_list)
    # print(model_str_list)

    image_size = 64
    nc = 3
    n_filters = 64
    after_conv = conv_size_comp(image_size)

    vae = Model(nz, nc, n_filters, after_conv)
    vae.load_state_dict(torch.load(os.path.join(savepath_prefix, 'infoVAE', 'checkpoint.pt')))
    if use_cuda:
        vae = vae.cuda(gpu_id)
    vae.eval()

    with torch.no_grad():
        for model_str in model_str_list:
            latent_space.main(savepath_prefix, nz, model_str, vae, gpu_id, use_cuda=True)



if __name__ == "__main__":
    gpu_id = 7
    nz = 3
    savepath_prefix = 'results/' + str(nz) + '-dims'
    model_str_dict = {'AGNrt': 0.9, 'NOAGNrt': 0.9, 'TNG100': 0.8, 'TNG50': 0.9, 'UHDrt': 1.0, 'n80rt': 1.0}
    model_str_list = ['AGNrt', 'NOAGNrt', 'TNG100', 'TNG50', 'UHDrt', 'n80rt']
    tsne_vis(savepath_prefix, nz, model_str_dict, model_str_list)
    latent_space_vis(savepath_prefix, gpu_id, nz, model_str_list)