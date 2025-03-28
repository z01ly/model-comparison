import os
import torch
import time
import yaml

import src.infoVAE.utils as utils
from src.infoVAE.mmdVAE import Model
import src.infoVAE.mmdVAE_train as mmdVAE_train
import src.infoVAE.mmdVAE_test as mmdVAE_test
import src.infoVAE.plot as plot


def vae(savepath_prefix):
    with open('src/infoVAE/infovae.yaml', 'r') as f:
        config = yaml.safe_load(f)

    for i in ['samples', 'loss', 'loss-plot']:
        os.makedirs(os.path.join(savepath_prefix, 'infoVAE', i), exist_ok=True)

    train_dataroot = 'data/sdss_data/train'
    train_dataloader = utils.dataloader_func(train_dataroot, 
                                            config['data_params']['train_batch_size'], 
                                            config['data_params']['num_workers'], 
                                            False)

    esval_dataroot = 'data/sdss_data/esval'
    esval_dataloader = utils.dataloader_func(esval_dataroot, 
                                            config['data_params']['val_batch_size'], 
                                            config['data_params']['num_workers'], 
                                            False)
    
    start_time = time.time()
    print("mmdvae training...\n")
    train_losses, val_losses, avg_train_losses, avg_val_losses = mmdVAE_train.train(savepath_prefix=savepath_prefix,
                                                                                    train_dataloader=train_dataloader,
                                                                                    val_dataloader=esval_dataloader,
                                                                                    use_cuda=True)
    end_time = time.time()

    elapsed_time = end_time - start_time
    with open(os.path.join(savepath_prefix, 'infoVAE', 'output.txt'), "a") as text_file:
        text_file.write(f"Training time: {elapsed_time:.6f} seconds. \n")  
    
    utils.save_losses(savepath_prefix, train_losses, val_losses, avg_train_losses, avg_val_losses)



def encoder(savepath_prefix, model_str_list):
    for i in ['train', 'test', 'sdss']:
        os.makedirs(os.path.join(savepath_prefix, 'latent-vectors', i), exist_ok=True)
        os.makedirs(os.path.join(savepath_prefix, 'dense-vectors', i), exist_ok=True)

    vae_save_path = os.path.join(savepath_prefix, 'infoVAE', 'checkpoint.pt')

    itr_tuple_list = [(model_str_list, 'data/mock_train', 'train'),
                      (model_str_list, 'data/mock_test', 'test'),
                      (['test'], 'data/sdss_data', 'sdss')]
    for str_tuple in itr_tuple_list:
        print(str_tuple)
        mmdVAE_test.test_main(str_tuple[0],
                              vae_save_path,
                              str_tuple[1],
                              os.path.join(savepath_prefix, 'latent-vectors', str_tuple[2]),
                              os.path.join(savepath_prefix, 'dense-vectors', str_tuple[2]))



def plot_training(savepath_prefix, es_pos, y_avg, y_itr):
    # training losses
    train_losses, val_losses, avg_train_losses, avg_val_losses = utils.load_losses(savepath_prefix)
    plot.plot_avg_loss(savepath_prefix, avg_train_losses, avg_val_losses, es_pos, y_avg)
    plot.plot_itr_loss(savepath_prefix, train_losses, val_losses, y_itr)



def plot_residual(savepath_prefix, config, model_str_list, use_cuda=True):
    os.makedirs(os.path.join(savepath_prefix, 'infoVAE', 'residual-plot'), exist_ok=True)

    vae = Model(config)
    vae.load_state_dict(torch.load(os.path.join(savepath_prefix, 'infoVAE', 'checkpoint.pt')))
    if use_cuda:
        vae = vae.cuda(config['trainer_params']['gpu_id'])
    vae.eval()

    with torch.no_grad():
        for model_str in model_str_list:
            for folder_path in [os.path.join('data/mock_train', model_str, 'test'), os.path.join('data/mock_test', model_str, 'test')]:
                plot.residual(vae, savepath_prefix, 1, model_str, folder_path, config, use_cuda=True)
        
        folder_path = 'data/sdss_data/test/cutouts'
        plot.residual(vae, savepath_prefix, 2, 'sdss-test', folder_path, config, use_cuda=True)
    


if __name__ == '__main__':
    gpu_id = 4
    nz = 32
    savepath_prefix = 'results/' + str(nz) + '-dims'
    model_str_list = ['AGNrt', 'NOAGNrt', 'TNG100', 'TNG50', 'UHDrt', 'n80rt']

    # vae(savepath_prefix, gpu_id, nz) # dim 3, 4, 16, 20 and 32 use default batch size 400
    # vae(savepath_prefix, gpu_id, nz, batch_size=400) # dim 2, try several times

    # encoder(savepath_prefix, model_str_list, gpu_id, nz)
    
    # plot_training(savepath_prefix, 36, 0.0025, 0.004) # es point: 16dim-31, 20dim-26, 32dim-40
    # plot_training(savepath_prefix, 23, 0.015, 0.015) # es point: 4dim-23
    # plot_training(savepath_prefix, 21, 0.02, 0.02) # es point: 2dim-13, 3dim-21
    
    # plot_residual(savepath_prefix, gpu_id, nz, model_str_list, use_cuda=True)
