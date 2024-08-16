# Adapted from https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb
# and https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# and https://github.com/pratikm141/MMD-Variational-Autoencoder-Pytorch-InfoVAE/blob/master/mmd_vae_pytorchver.ipynb

import torch
from torch import nn
from torch import Tensor

from src.infoVAE.utils import apply_k_sparse


# utils layer
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    

class Reshape(nn.Module):
    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape
    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)


# Encoder and decoder use the DC-GAN architecture
class Encoder(nn.Module):
    def __init__(self, z_dim, nc):
        super(Encoder, self).__init__()
        self.model = nn.ModuleList([
            # nn.Conv2d(nc, 32, 4, 2, 1),
            # nn.LeakyReLU(),
            nn.Conv2d(nc, 64, 3, 2, 1), 
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(),
            # nn.Conv2d(128, 256, 3, 2, 1),
            # nn.LeakyReLU(),
            # nn.Conv2d(256, 512, 4, 2, 1),
            # nn.LeakyReLU(),
            Flatten(),
            nn.Linear(128 * 16 * 16, 1024),
            nn.LeakyReLU(),
            # nn.Linear(1024, z_dim)
        ])

        self.fc_mu = nn.Linear(1024, z_dim)
        self.fc_var = nn.Linear(1024, z_dim)
        
    def forward(self, x):
        for layer in self.model:
            x = layer(x)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return [mu, log_var]
    

class Decoder(nn.Module):
    def __init__(self, z_dim, nc):
        super(Decoder, self).__init__()
        self.model = nn.ModuleList([
            nn.Linear(z_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * 16 * 16),
            nn.ReLU(),
            Reshape((128, 16, 16,)),
            # nn.ConvTranspose2d(512, 256, 4, 2, 1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(256, 128, 3, 2, 1),
            # nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, nc, 3, 2, 1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(32, nc, 4, 2, 1),
            nn.Sigmoid()
        ])
        
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x


class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(UpsampleConv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


# https://distill.pub/2016/deconv-checkerboard/
class NewDecoder(nn.Module):
    def __init__(self, z_dim, nc):
        super(NewDecoder, self).__init__()
        self.model = nn.ModuleList([
            nn.Linear(z_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * 16 * 16),
            nn.ReLU(),
            Reshape((128, 16, 16,)),
            # UpsampleConv(256, 128, scale_factor=2),
            # nn.ReLU(),
            UpsampleConv(128, 64, scale_factor=2),
            nn.ReLU(),
            UpsampleConv(64, nc, scale_factor=2),
            nn.Sigmoid()
        ])
    
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x


# model
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.encoder = Encoder(config['model_params']['latent_dim'], config['model_params']['in_channels'])
        self.decoder = NewDecoder(config['model_params']['latent_dim'], config['model_params']['in_channels'])
        
        # self._initialize_weights()

    # def _initialize_weights(self):
    #     def weights_init(m):
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
    #             nn.init.ones_(m.weight)
    #             nn.init.zeros_(m.bias)
    #     
    #     self.encoder.apply(weights_init)
    #     self.decoder.apply(weights_init)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
        
    def forward(self, x, k_pre_value):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)

        z_sparse = apply_k_sparse(z, k_pre=k_pre_value)

        x_reconstructed = self.decoder(z_sparse)

        return z_sparse, x_reconstructed, mu, log_var, z 
    
    def compute_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
        return torch.exp(-kernel_input) # (x_size, y_size)
    
    def compute_mmd(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
        return mmd

    def loss_func(self, true_samples, z, x_reconstructed, x, mu, log_var):
        # kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0) # VAE kl divergence

        # rho_hat = torch.mean(z, dim=0)
        # rho = 0.05
        # sparse_kld = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        # sparse_kld = sparse_kld.sum()

        mmd = self.compute_mmd(true_samples, z)
        nll = (x_reconstructed - x).pow(2).mean() # reconstruction loss
        loss = nll + mmd

        return {'loss': loss, 'nll': nll, 'mmd': mmd}



if __name__ == "__main__":
    import yaml
    with open('src/infoVAE/infovae.yaml', 'r') as f:
        config = yaml.safe_load(f)
    for epoch in range(10):
        k_pre_value = config['model_params']['k_pre_value'] + config['model_params']["k_step"] * (10 - epoch)
        print(k_pre_value)
