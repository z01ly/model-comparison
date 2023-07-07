import torch
from torch.autograd import Variable

from torchvision import transforms, datasets

import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

import utils
from mmdVAE_train import Model

def test(dataloader, z_dim=2, nc=3, n_filters=64, after_conv=16, use_cuda=True):
    model = Model(z_dim, nc, n_filters, after_conv)
    if use_cuda:
        model = model.cuda(gpu_id)
    model.load_state_dict(torch.load('./sdss_model_param/model_weights.pth'))
    model.eval()

    z_list = []
    for batch_idx, (test_x, _) in enumerate(dataloader):
        test_x = Variable(test_x, requires_grad=False)
        if(use_cuda):
            test_x = test_x.cuda(gpu_id)

        z = model.encoder(test_x)
        z_list.append(z.cpu().data.numpy())
         
    z = np.concatenate(z_list, axis=0)

    return z



if __name__ == "__main__":
    gpu_id = 6
    workers = 4
    batch_size = 500
    image_size = 64
    nc = 3
    nz = 32 # Size of z latent vector
    n_filters = 64
    after_conv = utils.conv_size_comp(image_size)

    test_dataroot_1 = '../NOAGN'
    test_dataset_1 = datasets.ImageFolder(root=test_dataroot_1,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                ]))

    test_dataloader_1 = torch.utils.data.DataLoader(test_dataset_1, batch_size=batch_size,
                                            shuffle=True, num_workers=workers,
                                            pin_memory=True)

    z_1 = test(dataloader=test_dataloader_1, z_dim=nz,
                nc=nc, n_filters=n_filters, after_conv=after_conv)
    print(z_1.shape)
    # with open("./test_result_txts/NOAGN.txt", "w") as output:
    #     output.write(str(z_1))
