import torch
from torch.autograd import Variable

from torchvision import transforms, datasets

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

import src.infoVAE.utils as utils
from src.infoVAE.mmdVAE_train import Model


def test(model, test_dataroot, savefig_path, z_dim=2, 
        nc=3, n_filters=64, after_conv=16, use_cuda=True, 
        gpu_id=0, workers=4, batch_size=500):

    # dataloader
    dataloader = utils.dataloader_func(test_dataroot, batch_size, workers, True)

    z_list = []
    for batch_idx, (test_x, _) in enumerate(dataloader):
        test_x = Variable(test_x, requires_grad=False)
        if(use_cuda):
            test_x = test_x.cuda(gpu_id)

        z = model.encoder(test_x)
        z_list.append(z.cpu().data.numpy())
         
    z = np.concatenate(z_list, axis=0)

    # 16 images
    sampled_rows = np.random.choice(z.shape[0], size=16, replace=False)
    sampled_matrix = z[sampled_rows]
    gen_z = Variable(torch.tensor(sampled_matrix), requires_grad=False)
    if use_cuda:
        gen_z = gen_z.cuda(gpu_id)
    samples = model.decoder(gen_z)
    samples = samples.permute(0,2,3,1).contiguous().cpu().data.numpy()
    plt.imshow(utils.convert_to_display(samples))
    plt.savefig(savefig_path, dpi=300)
    
    return z



def sdss_test_with_filename(model, z_dim=2, nc=3, n_filters=64, 
        after_conv=16, use_cuda=True, gpu_id=0, workers=4, batch_size=500):
    
    dataloader = utils.dataloader_func('src/data/sdss_data/test', batch_size, workers, True, with_filename=True)

    z_list = []
    filename_list = []
    for batch_idx, (test_x, _, img_filenames) in enumerate(dataloader):
        test_x = Variable(test_x, requires_grad=False)
        if(use_cuda):
            test_x = test_x.cuda(gpu_id)

        z = model.encoder(test_x)

        z_list.append(z.cpu().data.numpy())
        filename_list.extend(img_filenames)
    
    # z[idx] corresponds to filename_arr[idx]
    z = np.concatenate(z_list, axis=0)
    filename_arr = np.asarray(filename_list)

    return z, filename_arr





if __name__ == "__main__":
    gpu_id = 1
    workers = 4
    batch_size = 500 # CUDA out of memory if 500x500 image size
    image_size = 64 # downsample 500x500 test images to 64x64
    nc = 3
    nz = 32 # Size of z latent vector
    z_dim = nz
    n_filters = 64
    after_conv = utils.conv_size_comp(image_size)
    use_cuda = True

    # model
    model = Model(z_dim, nc, n_filters, after_conv)
    model.load_state_dict(torch.load('src/infoVAE/mmdVAE_save/checkpoint.pt'))
    if use_cuda:
        model = model.cuda(gpu_id)
    model.eval()

    with torch.no_grad():
        # test_dataroots = [os.path.join('src/data/mock_trainset', subdir) for subdir in os.listdir('src/data/mock_trainset')]
        # test_dataroots.extend([os.path.join('src/data/mock_valset', subdir) for subdir in os.listdir('src/data/mock_valset')])

        # test_dataroots = [os.path.join('src/data/mock_trainset', subdir) for subdir in ['UHD_2times', 'n80_2times', 'TNG50-1_snapnum_099_2times']]
        test_dataroots = [os.path.join('src/data/mock_trainset', subdir) for subdir in ['mockobs_0915_2times']]
        for test_dataroot in test_dataroots:
            directory_names = test_dataroot.split(os.path.sep)
            extraction = f"{directory_names[-2][5: ]}_{directory_names[-1]}"

            savefig_path = 'src/infoVAE/test_results/images_in_testing/fig_' + extraction + '.png'
            z = test(model, test_dataroot=test_dataroot, savefig_path=savefig_path,
                    z_dim=nz, nc=nc, n_filters=n_filters, after_conv=after_conv, 
                    use_cuda=True, gpu_id=gpu_id, workers=workers, batch_size=batch_size)

            np.save('src/infoVAE/test_results/latent/' + extraction + '.npy', z)
        
        

        """
        sdss_test_dataroot = 'src/data/sdss_data/test'
        savefig_path = 'src/infoVAE/test_results/images_in_testing/fig_sdss_test.png'
        z = test(model, test_dataroot=sdss_test_dataroot, savefig_path=savefig_path,
                z_dim=nz, nc=nc, n_filters=n_filters, after_conv=after_conv, 
                use_cuda=True, gpu_id=gpu_id, workers=workers, batch_size=batch_size)
        np.save('src/infoVAE/test_results/latent/sdss_test.npy', z)
        """

        """
        z, filename_arr = sdss_test_with_filename(model, 
                z_dim=nz, nc=nc, n_filters=n_filters, after_conv=after_conv, 
                use_cuda=True, gpu_id=gpu_id, workers=workers, batch_size=batch_size)
        np.save('src/infoVAE/test_results/latent/sdss_test.npy', z)
        np.save('src/infoVAE/test_results/latent/sdss_test_filenames.npy', filename_arr)
        """
    

    for filename_latent in os.listdir('src/infoVAE/test_results/latent/'):
        if filename_latent[-3: ] == 'npy':
            latent_z = np.load('src/infoVAE/test_results/latent/' + filename_latent)
            print(latent_z.shape)
    print('\n')


    