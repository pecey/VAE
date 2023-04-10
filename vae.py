import torch as T
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, in_dim, latent_dim, device = "cpu"):
        super(VAE, self).__init__()
        self.h, self.w = in_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_1 = nn.Conv2d(3, 16, 3, stride = 2, padding = 1)
        self.encoder_2 = nn.Conv2d(16, 16, 3, stride = 2, padding = 1)

        # Latent Space
        self.latent_layer_mean = nn.Linear(16 * 16 * 16, latent_dim)
        self.latent_layer_variance = nn.Linear(16 * 16 * 16, latent_dim)
        self.latent_decoder = nn.Linear(latent_dim, 16 * 16 * 16)

        # Decoder
        self.decoder_1 = nn.ConvTranspose2d(16 * 16 * 16, 16, 16, stride = 2)
        self.decoder_2 = nn.ConvTranspose2d(16, 16, 3, stride = 2)
        self.decoder_3 = nn.ConvTranspose2d(16, 3, 3, stride = 2)
       
        self.to(device)

    def encode(self, x):
        x = F.relu(self.encoder_1(x))
        x = F.relu(self.encoder_2(x))
        x = x.reshape(-1, 16 * 16 * 16)
        return self.latent_layer_mean(x), self.latent_layer_variance(x)

    def reparameterization(self, mean, variance):
        std = T.exp(0.5 * variance)
        eps = T.rand_like(std)
        return mean + std * eps

    def decode(self, x):
        x = self.latent_decoder(x).reshape(-1, 16 * 16 * 16, 1, 1)
        x = F.relu(self.decoder_1(x))
        x = F.relu(self.decoder_2(x)[:, :, :32, :32])
        x = T.sigmoid(self.decoder_3(x)[:, :, :64, :64])
        return x

    def forward(self, x):
        mean, variance = self.encode(x)
        z = self.reparameterization(mean, variance)
        return self.decode(z), mean, variance