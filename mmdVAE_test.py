import torch
from torch.autograd import Variable

from torchvision import transforms, datasets

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils
from mmdVAE_train import Model

def test(test_dataroot, savefig_path, z_dim=2, 
        nc=3, n_filters=64, after_conv=16, use_cuda=True, 
        gpu_id=0, workers=4, batch_size=500):
    # model
    model = Model(z_dim, nc, n_filters, after_conv)
    model.load_state_dict(torch.load('./mmdVAE_save/checkpoint.pt'))
    if use_cuda:
        model = model.cuda(gpu_id)
    model.eval()

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

    # 25 images
    sampled_rows = np.random.choice(z.shape[0], size=25, replace=False)
    sampled_matrix = z[sampled_rows]
    gen_z = Variable(torch.tensor(sampled_matrix), requires_grad=False)
    if use_cuda:
        gen_z = gen_z.cuda(gpu_id)
    samples = model.decoder(gen_z)
    samples = samples.permute(0,2,3,1).contiguous().cpu().data.numpy()
    plt.imshow(utils.convert_to_display(samples))
    plt.savefig(savefig_path, dpi=300)
    
    return z


if __name__ == "__main__":
    gpu_id = 4
    workers = 4
    batch_size = 500 # CUDA out of memory if 500x500 image size
    image_size = 64 # downsample 500x500 test images to 64x64
    nc = 3
    nz = 32 # Size of z latent vector
    n_filters = 64
    after_conv = utils.conv_size_comp(image_size)

    # z_NOAGN = test(test_dataroot='../NOAGN', savefig_path='./test_results/fig_NOAGN.png',
    #             z_dim=nz, nc=nc, n_filters=n_filters, after_conv=after_conv, 
    #             use_cuda=True, gpu_id=gpu_id, workers=4, batch_size=500)

    # z_AGN = test(test_dataroot='../AGN', savefig_path='./test_results/fig_AGN.png',
    #             z_dim=nz, nc=nc, n_filters=n_filters, after_conv=after_conv, 
    #             use_cuda=True, gpu_id=gpu_id, workers=4, batch_size=500)

    # z_n80 = test(test_dataroot='../n80', savefig_path='./test_results/fig_n80.png',
    #             z_dim=nz, nc=nc, n_filters=n_filters, after_conv=after_conv, 
    #             use_cuda=True, gpu_id=gpu_id, workers=4, batch_size=500)

    z_UHD = test(test_dataroot='../UHD', savefig_path='./test_results/fig_UHD.png',
                z_dim=nz, nc=nc, n_filters=n_filters, after_conv=after_conv, 
                use_cuda=True, gpu_id=gpu_id, workers=4, batch_size=500)
    
    print(z_UHD.shape)
    # with open("./test_results/NOAGN.txt", "w") as output:
    #     output.write(str(z_1.shape))

    
