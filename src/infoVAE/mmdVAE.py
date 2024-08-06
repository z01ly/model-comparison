# Adapted from https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb
# and https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# and https://github.com/pratikm141/MMD-Variational-Autoencoder-Pytorch-InfoVAE/blob/master/mmd_vae_pytorchver.ipynb

import torch
from torch import nn
from torch import Tensor

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
            # nn.Conv2d(nc, 32, 4, 2, 1), # 32, 32, 32
            # nn.LeakyReLU(),
            nn.Conv2d(nc, 64, 4, 2, 1), # 64, 16, 16
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 128, 8, 8
            nn.LeakyReLU(),
            # nn.Conv2d(128, 256, 4, 2, 1), # 256, 4, 4
            # nn.LeakyReLU(),
            # nn.Conv2d(256, 512, 4, 2, 1), # 512, 2, 2
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
            # nn.ConvTranspose2d(256, 128, 4, 2, 1),
            # nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, nc, 4, 2, 1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(32, nc, 4, 2, 1),
            nn.Sigmoid()
        ])
        
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x



# model
class Model(nn.Module):
    def __init__(self, z_dim, nc):
        super(Model, self).__init__()
        self.encoder = Encoder(z_dim, nc)
        self.decoder = Decoder(z_dim, nc)

        self.z_dim = z_dim

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
    
    def k_sparse_constraint(self, z, k):
        assert k <= z.shape[1], "k should not exceed the latent dimension"

        # Get the top-k values and their indices along the feature dimension
        topk_values, _ = torch.topk(z, k, dim=1, largest=True, sorted=False)
        # Find the minimum value among the top-k values in each sample
        min_topk_values = topk_values[:, -1].view(-1, 1)
        # Create a mask where values are greater than or equal to the minimum top-k value
        mask = z >= min_topk_values
        # print(mask)

        # Apply the mask to the activations, setting all but the top k to zero
        z_sparse = z * mask.float()
        
        return z_sparse
        
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)

        z_sparse = self.k_sparse_constraint(z, k=int(self.z_dim / 2))

        x_reconstructed = self.decoder(z_sparse)

        return z, x_reconstructed, mu, log_var
    
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

    def loss_func(self, config, true_samples, z, x_reconstructed, x, mu, log_var):
        # kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0) # VAE kl divergence

        # rho_hat = torch.mean(z, dim=0)
        # rho = 0.05
        # sparse_kld = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        # sparse_kld = sparse_kld.sum()

        mmd = self.compute_mmd(true_samples, z)
        nll = (x_reconstructed - x).pow(2).mean() # reconstruction loss
        loss = nll + mmd

        return {'loss': loss, 'nll': nll, 'mmd': mmd}

