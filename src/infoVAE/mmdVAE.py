# Adapted from https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb
# and https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# and https://github.com/pratikm141/MMD-Variational-Autoencoder-Pytorch-InfoVAE/blob/master/mmd_vae_pytorchver.ipynb

import torch

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
    def __init__(self, z_dim, nc, n_filters, after_conv):
        super(Decoder, self).__init__()
        self.model = torch.nn.ModuleList([
            torch.nn.Linear(z_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, after_conv * after_conv * 2 * n_filters),
            torch.nn.ReLU(),
            Reshape((2 * n_filters, after_conv, after_conv,)),
            torch.nn.ConvTranspose2d(n_filters * 2, n_filters, 4, 2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(n_filters, nc, 4, 2, padding=1),
            torch.nn.Sigmoid()
        ])
        
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x



# model
class Model(torch.nn.Module):
    def __init__(self, z_dim, nc, n_filters, after_conv):
        super(Model, self).__init__()
        self.encoder = Encoder(z_dim, nc, n_filters, after_conv)
        self.decoder = Decoder(z_dim, nc, n_filters, after_conv)
        
    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return z, x_reconstructed
