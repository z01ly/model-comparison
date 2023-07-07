# Adapted from https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb
# and https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# and https://github.com/pratikm141/MMD-Variational-Autoencoder-Pytorch-InfoVAE/blob/master/mmd_vae_pytorchver.ipynb

import torch
from torch.autograd import Variable

from torchvision import transforms, datasets

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math, os

import utils

# utils layer
class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    

class Reshape(torch.nn.Module):
    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape
    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)


# Encoder and decoder use the DC-GAN architecture
class Encoder(torch.nn.Module):
    def __init__(self, z_dim, nc, n_filters, after_conv):
        super(Encoder, self).__init__()
        self.model = torch.nn.ModuleList([
            torch.nn.Conv2d(nc, n_filters, 4, 2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(n_filters, n_filters * 2, 4, 2, padding=1),
            torch.nn.LeakyReLU(),
            Flatten(),
            torch.nn.Linear(n_filters * 2 * after_conv * after_conv, 1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, z_dim)
        ])
        
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x
    

class Decoder(torch.nn.Module):
    def __init__(self, z_dim, n_filters, after_conv):
        super(Decoder, self).__init__()
        self.model = torch.nn.ModuleList([
            torch.nn.Linear(z_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, after_conv * after_conv * 2 * n_filters),
            torch.nn.ReLU(),
            Reshape((2 * n_filters, after_conv, after_conv,)),
            torch.nn.ConvTranspose2d(n_filters * 2, n_filters, 4, 2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(n_filters, 1, 4, 2, padding=1),
            torch.nn.Sigmoid()
        ])
        
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x


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


# model
class Model(torch.nn.Module):
    def __init__(self, z_dim, nc, n_filters, after_conv):
        super(Model, self).__init__()
        self.encoder = Encoder(z_dim, nc, n_filters, after_conv)
        self.decoder = Decoder(z_dim, n_filters, after_conv)
        
    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return z, x_reconstructed


# Convert a numpy array of shape [batch_size, height, width, 1] into a displayable array 
# of shape [height*sqrt(batch_size), width*sqrt(batch_size))] by tiling the images
def convert_to_display(samples):
    cnt, height, width = int(math.floor(math.sqrt(samples.shape[0]))), samples.shape[1], samples.shape[2]
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height, cnt, cnt, width])
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height*cnt, width*cnt])
    return samples


# Training
def train(dataloader, z_dim=2, nc=3, n_filters=64, after_conv=16, n_epochs=10, use_cuda=True, gpu_id=0):
    model = Model(z_dim, nc, n_filters, after_conv)
    if use_cuda:
        model = model.cuda(gpu_id)

    optimizer = torch.optim.Adam(model.parameters())

    # i = -1
    for epoch in range(n_epochs):
        for batch_idx, (images, _) in enumerate(dataloader):
            # i += 1
            optimizer.zero_grad()
            x = Variable(images, requires_grad=False)
            # sample: len(images)
            true_samples = Variable(torch.randn(len(images), z_dim), requires_grad=False)
            if use_cuda:
                x = x.cuda(gpu_id)
                true_samples = true_samples.cuda(gpu_id)
            
            z, x_reconstructed = model(x)
            mmd = compute_mmd(true_samples, z)
            nll = (x_reconstructed - x).pow(2).mean()
            loss = nll + mmd

            loss.backward()

            optimizer.step()

            # nll.data[0], mmd.data[0]: invalid index of a 0-dim tensor
            # if i % print_every == 0:
                # print("Negative log likelihood is {:.5f}, mmd loss is {:.5f}".format(nll.data, mmd.data))
        text = "Negative log likelihood is {:.5f}, mmd loss is {:.5f}. epoch {} \n".format(nll.data, mmd.data, epoch)
        with open("./sdss_model_param/loss.txt", "a") as loss_file:
            loss_file.write(text)

        # 100 images
        gen_z = Variable(torch.randn(100, z_dim), requires_grad=False)
        if use_cuda:
            gen_z = gen_z.cuda(gpu_id)
        samples = model.decoder(gen_z)
        samples = samples.permute(0,2,3,1).contiguous().cpu().data.numpy()
        # colormap: 'plasma', 'cubehelix' and 'jet' are all ok
        plt.imshow(convert_to_display(samples), cmap='plasma')
        savefig_path = './sdss_model_param/fig_epoch' + str(epoch) + '.png' 
        plt.savefig(savefig_path, dpi=300)
        # plt.show()

    return model


# Main
if __name__ == "__main__":
    gpu_id = 1
    workers = 4
    batch_size = 200
    image_size = 64 # sdss image size
    nc = 3 # Number of channels in the training images. For color images this is 3
    nz = 32 # Size of z latent vector
    n_filters = 64 # Size of feature maps
    num_epochs = 10 # Number of training epochs
    after_conv = utils.conv_size_comp(image_size)

    train_dataroot = '../sdss'
    train_dataset = datasets.ImageFolder(root=train_dataroot,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                ]))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)


    model = train(dataloader=train_dataloader, z_dim=nz,
                nc=nc, n_filters=n_filters, after_conv=after_conv, n_epochs=num_epochs, 
                use_cuda=True, gpu_id=gpu_id)

    torch.save(model.state_dict(), './sdss_model_param/model_weights.pth')


