import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from code import vqvae
from code.utils import HParams, get_param_count


_ACTIVATION_MAP = {
    'relu': F.relu,
    'gelu': F.gelu,
    'elu': F.elu,
    'selu': F.selu,
}


_HP_BASE = HParams(
    # parameters to be specified at model creation
    image_shape         = None,
    codebook_bits       = None,
    ch_in               = None,
    ch_latent           = None,
    latent_shape        = None,
    enc_si              = None,
    dec_si              = None,

    # VQVAE architecture
    embedding_dim       = 64,
    cond_ch_in          = 32,
    ch_hidden           = 32,
    ch_residual         = 32,
    num_residual_layers = 3,
    activation          = 'gelu',
    decay               = 0.99,
    eps                 = 1e-5,

    # Training
    batch_size          = 32,       # distributed across num_gpus
    learning_rate       = 5e-4,
    beta                = 0.25,     # commitment cost
    max_epoch           = 20,

    # Monitoring
    print_freq          = 10,
    log_freq            = 50,
    sample_freq         = 1,
    ckpt_freq           = 5,
)


class Encoder(nn.Module):
    def __init__(self, ch_in, ch_hidden, ch_residual, num_residual_layers, activation, cond_ch_in: int = None):
        super().__init__()

        self.conv_in = vqvae.Conv2d(ch_in, ch_hidden // 2, kernel_size=3, stride=1, padding=1)
        if cond_ch_in is None:
            self.has_sideinfo = False
            self.cond_conv_in = None
        else:
            self.has_sideinfo = True
            self.cond_conv_in = vqvae.Conv2d(cond_ch_in, ch_hidden // 2, kernel_size=3, stride=1, padding=1)

        self.res1 = vqvae.ResidualBlock(1 * ch_hidden, ch_residual, num_residual_layers, activation)
        self.res2 = vqvae.ResidualBlock(2 * ch_hidden, ch_residual, num_residual_layers, activation)
        self.res3 = vqvae.ResidualBlock(4 * ch_hidden, ch_residual, num_residual_layers, activation)
        self.conv1 = vqvae.Conv2d(1 * ch_hidden, 2 * ch_hidden, kernel_size=4, stride=2, padding=1)
        self.conv2 = vqvae.Conv2d(2 * ch_hidden, 4 * ch_hidden, kernel_size=4, stride=2, padding=1)
        self.conv3 = vqvae.Conv2d(4 * ch_hidden, 8 * ch_hidden, kernel_size=4, stride=2, padding=1)


    def forward(self, x, c=None):
        x = self.conv_in(x)
        if self.has_sideinfo:
            c = self.cond_conv_in(c)
        else:
            c = torch.zeros_like(x)

        xc = torch.cat([x, c], dim=1)
        xc = self.conv1(self.res1(xc))          
        xc = self.conv2(self.res2(xc))          
        xc = self.conv3(self.res3(xc))          
        return xc

    def get_non_si_param_count(self):
        param_count = get_param_count(self)
        if self.has_sideinfo:
            param_count -= get_param_count(self.cond_conv_in)
        return param_count


class Decoder(nn.Module):
    def __init__(self, ch_in, ch_hidden, ch_out, ch_residual, num_residual_layers, activation, cond_ch_in: int = None):
        super().__init__()

        act_fn = _ACTIVATION_MAP[activation]
        self.conv_in = vqvae.Conv2d(ch_in, 4 * ch_hidden, kernel_size=3, stride=1, padding=1)
        if cond_ch_in is None:
            self.has_sideinfo = False
            self.cond_conv_in = None
        else:
            self.has_sideinfo = True
            # self.cond_conv_in = vqvae.Conv2d(cond_ch_in, 4 * ch_hidden, kernel_size=3, stride=1, padding=1)
            # Mimic the encoder
            self.cond_conv_in = nn.Sequential(
                vqvae.Conv2d(cond_ch_in, ch_hidden, kernel_size=3, stride=1, padding=1),
                vqvae.ResidualBlock(1 * ch_hidden, ch_residual, num_residual_layers, activation),
                vqvae.Conv2d(1 * ch_hidden, 2 * ch_hidden, kernel_size=4, stride=2, padding=1),
                vqvae.ResidualBlock(2 * ch_hidden, ch_residual, num_residual_layers, activation),
                vqvae.Conv2d(2 * ch_hidden, 4 * ch_hidden, kernel_size=4, stride=2, padding=1),
                vqvae.ResidualBlock(4 * ch_hidden, ch_residual, num_residual_layers, activation),
                vqvae.Conv2d(4 * ch_hidden, 4 * ch_hidden, kernel_size=4, stride=2, padding=1),
            )

        self.tconv1 = vqvae.ConvTranspose2d(8 * ch_hidden, 4 * ch_hidden, kernel_size=4, stride=2, padding=1)
        self.tconv2 = vqvae.ConvTranspose2d(4 * ch_hidden, 2 * ch_hidden, kernel_size=4, stride=2, padding=1)
        self.tconv3 = vqvae.ConvTranspose2d(2 * ch_hidden, 1 * ch_hidden, kernel_size=4, stride=2, padding=1)
        self.res1 = vqvae.ResidualBlock(4 * ch_hidden, ch_residual, num_residual_layers, activation)
        self.res2 = vqvae.ResidualBlock(2 *ch_hidden, ch_residual, num_residual_layers, activation)
        self.res3 = vqvae.ResidualBlock(1 *ch_hidden, ch_residual, num_residual_layers, activation)
        self.cond_out = vqvae.Conv2d(1 * ch_hidden, ch_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x, c=None):
        x = self.conv_in(x)
        if self.has_sideinfo:
            c = self.cond_conv_in(c)
        else:
            c = torch.zeros_like(x)

        xc = torch.cat([x, c], dim=1)
        xc = self.res1(self.tconv1(xc))
        xc = self.res2(self.tconv2(xc))
        xc = self.res3(self.tconv3(xc))
        xc = self.cond_out(xc)
        xc = torch.sigmoid(xc) - 0.5

        return xc

    def get_non_si_param_count(self):
        param_count = get_param_count(self)
        if self.has_sideinfo:
            param_count -= get_param_count(self.cond_conv_in)
        return param_count


class SINet(nn.Module):
    def __init__(self, ch_in, ch_hidden, ch_residual, num_residual_layers, activation):
        super().__init__()
        self.conv_in = vqvae.Conv2d(ch_in, ch_hidden, kernel_size=3, stride=1, padding=1)
        self.res = vqvae.ResidualBlock(ch_hidden, ch_residual, num_residual_layers, activation)

    def forward(self, x):
        x = self.res(self.conv_in(x))
        return x


class VqvaeTop8x(nn.Module):
    def __init__(self, *,
                 image_shape,  # full image shape, e.g. (3, 64, 64)
                 codebook_bits: int,
                 ch_latent: int,
                 enc_si: bool,
                 dec_si: bool,
                 **kwargs):
        """
            Input     : top half    -> (N, C, H//2, W)
            Cond input: bottom half -> (N, C, H//2, W)
            Latent    : (N, embedding_dim, ch_latent, H//16, W//8)
              -> Number of latents: ch_latent * (H//16) * (W//8)
        """
        super().__init__()
        self.hp = _HP_BASE.clone()
        self.hp.image_shape = tuple(image_shape)
        self.hp.ch_in = image_shape[0]
        self.hp.codebook_bits = codebook_bits
        self.hp.ch_latent = ch_latent
        self.hp.enc_si = enc_si
        self.hp.dec_si = dec_si
        self.hp.latent_shape = (ch_latent, image_shape[1]//16, image_shape[2]//8)
        self.use_sideinfo = (enc_si or dec_si)

        for k, v in kwargs.items():
            assert k in self.hp, f'Invalid hparam {k} given'
            if v != self.hp[k]:
                print(f'Overriding hparam {k}: {v} (default: {self.hp[k]})')
                self.hp[k] = v

        # Modules
        self.encoder = Encoder(self.hp.ch_in, self.hp.ch_hidden, self.hp.ch_residual, self.hp.num_residual_layers,
                               self.hp.activation, cond_ch_in=(self.hp.cond_ch_in if enc_si else None))
        self.decoder = Decoder(self.hp.embedding_dim * self.hp.ch_latent, self.hp.ch_hidden, self.hp.ch_in, self.hp.ch_residual,
                               self.hp.num_residual_layers, self.hp.activation, cond_ch_in=(self.hp.cond_ch_in if dec_si else None))
        self.z_proj = vqvae.Conv2d(8 * self.hp.ch_hidden, self.hp.embedding_dim * self.hp.ch_latent, kernel_size=1, stride=1, padding=0)
        self.quantizer = vqvae.VectorQuantizerEMA(2**self.hp.codebook_bits, self.hp.embedding_dim, decay=self.hp.decay, eps=self.hp.eps)
        if enc_si or dec_si:
            self.si_net = SINet(self.hp.ch_in, self.hp.cond_ch_in, self.hp.ch_residual, self.hp.num_residual_layers, self.hp.activation)
        else:
            self.si_net = None

    def forward(self, x, c=None):
        # Process side info
        if self.use_sideinfo:
            assert c is not None
            c = self.si_net(c)
        else:
            c = None

        # Encoder + quantization
        z_enc = self.encoder(x, c=c)
        z_e = self.z_proj(z_enc)
        # assert z_e.shape[1:] == (self.hp.embedding_dim * self.hp.ch_latent, 8, 16)
        zs = z_e.shape
        z_e = z_e.view(zs[0], self.hp.embedding_dim, self.hp.ch_latent, zs[2], zs[3])
        z_q, emb, idx = self.quantizer(z_e)
        z_q = z_q.view(*zs)

        # Decoder
        x_rec = self.decoder(z_q, c=c)  # (N, C, H//2, W)
        return x_rec, z_q, emb, z_e.view(*zs), idx

    def decode_indices(self, indices, c=None):
        if self.use_sideinfo:
            c = self.si_net(c)

        assert indices.shape[1] == self.hp.ch_latent and indices.ndim == 4
        z_q = self.quantizer.embed(indices)
        z_q = z_q.view(z_q.shape[0], -1, z_q.shape[3], z_q.shape[4])
        x = self.decoder(z_q, c=c)
        return x

    def get_autoencoder_param_count(self):
        return self.encoder.get_non_si_param_count() + self.decoder.get_non_si_param_count() + get_param_count(self.z_proj) + self.quantizer.get_param_count()