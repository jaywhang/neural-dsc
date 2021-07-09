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
    dim                 = None,
    codebook_bits       = None,
    d_latent            = None,
    enc_si              = None,
    dec_si              = None,

    # VQVAE architecture
    embedding_dim       = 160,
    max_timestep        = 10000,
    d_hidden            = 256,
    d_sideinfo          = 412,
    d_residual          = 160,
    num_residual_layers = 4,
    activation          = 'gelu',
    decay               = 0.99,
    eps                 = 1e-5,

    # Training
    batch_size          = 128,       # distributed across num_gpus
    learning_rate       = 1e-4,
    beta                = 0.15,     # commitment cost
    max_epoch           = 500,

    # Monitoring
    print_freq          = 10,
    log_freq            = 50,
    ckpt_freq           = 100,
    eval_freq           = 5,
)


DIM = 412


class ResidualBlockFC(nn.Module):
    def __init__(self, dim: int, d_hidden: int, activation: str):
        super().__init__()
        act_class = {
            'relu': nn.ReLU,
            'elu': nn.ELU,
            'gelu': nn.GELU,
        }[activation]
        self.layers = nn.Sequential(
            act_class(),
            nn.Linear(dim, d_hidden),
            act_class(),
            nn.Linear(d_hidden, d_hidden),
            act_class(),
            nn.Linear(d_hidden, dim),
        )

    def forward(self, h):
        h = h + self.layers(h)
        return h


class EncoderFC(nn.Module):
    def __init__(self, *, dim: int, d_hidden: int, d_out: int, d_residual: int, d_cond: int = None):
        assert d_hidden % 2 == 0
        super().__init__()

        self.is_conditional = (d_cond is not None)
        self.d_hidden = d_hidden
        self.fc_in = nn.Linear(dim, d_hidden // 2)
        self.fc_cond = nn.Linear(dim, d_hidden // 2) if self.is_conditional else None
        self.res1 = ResidualBlockFC(d_hidden, d_residual, 'gelu')
        self.fc1 = nn.Linear(d_hidden, d_hidden)
        self.res2 = ResidualBlockFC(d_hidden, d_residual, 'gelu')
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x, t_emb, c=None):
        assert x.ndim == 2 and t_emb.shape == (len(x), self.d_hidden)
        x = self.fc_in(x)
        if self.is_conditional:
            c = self.fc_cond(c)
        else:
            c = torch.zeros_like(x)
        xc = torch.cat([x, c], dim=1) + t_emb
        assert xc.shape == (len(x), self.d_hidden)
        xc = self.res1(xc)
        xc = self.fc1(F.gelu(xc))
        xc = self.res2(xc)
        xc = self.fc2(F.gelu(xc))
        return xc

    def get_non_si_param_count(self):
        return -1
        # param_count = get_param_count(self)
        # if self.has_sideinfo:
        #     param_count -= get_param_count(self.cond_conv_in)
        # return param_count


class DecoderFC(nn.Module):
    def __init__(self, *, d_in: int, d_hidden: int, d_residual: int, d_out: int, d_cond: int = None):
        assert d_hidden % 2 == 0
        super().__init__()
        self.is_conditional = (d_cond is not None)
        self.d_hidden = d_hidden
        self.fc_in = nn.Linear(d_in, d_hidden // 2)
        self.fc_cond = nn.Linear(d_cond, d_hidden // 2) if self.is_conditional else None
        self.layers = nn.Sequential(
            ResidualBlockFC(d_hidden, d_residual, 'gelu'),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            ResidualBlockFC(d_hidden, d_residual, 'gelu'),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x, t_emb, c=None):
        assert x.ndim == 2 and t_emb.shape == (len(x), self.d_hidden)
        x = self.fc_in(x)
        if self.is_conditional:
            c = self.fc_cond(c)
        else:
            c = torch.zeros_like(x)
        xc = torch.cat([x, c], dim=1) + t_emb
        assert xc.shape == (len(x), self.d_hidden)
        xc = self.layers(xc)
        return xc

    def get_non_si_param_count(self):
        return -1
        # param_count = get_param_count(self)
        # if self.has_sideinfo:
        #     param_count -= get_param_count(self.cond_conv_in)
        # return param_count


class SINet(nn.Module):
    def __init__(self, d_in, d_out, d_residual, d_hidden, num_residual_layers, activation):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out

        self.fc = nn.Linear(d_in, d_out)
        self.fc_t = nn.Linear(d_hidden, d_out)
        self.res = ResidualBlockFC(d_out, d_residual, 'gelu')

    def forward(self, x, t_emb):
        assert x.shape == (len(x), self.d_in) and t_emb.shape == (len(x), self.d_hidden)
        x = self.fc(x)
        t = self.fc_t(t_emb)
        assert x.shape == t.shape == (len(x), self.d_out)
        x = self.res(x + t)
        return x


class VqvaeMnistGradTc(nn.Module):
    def __init__(self, *,
                 dim = DIM,
                 codebook_bits: int,
                 d_latent: int,
                 enc_si: bool,
                 dec_si: bool,
                 **kwargs):
        """
            Input     : (N, D)
            Cond input: (N, D)
            Latent    : (N, d_latent)
        """
        super().__init__()
        self.hp = _HP_BASE.clone()
        self.hp.dim = dim
        self.hp.codebook_bits = codebook_bits
        self.hp.d_latent = d_latent
        self.hp.enc_si = enc_si
        self.hp.dec_si = dec_si
        self.use_sideinfo = (enc_si or dec_si)

        for k, v in kwargs.items():
            assert k in self.hp, f'Invalid hparam {k} given'
            if v != self.hp[k]:
                print(f'Overriding hparam {k}: {v} (default: {self.hp[k]})')
                self.hp[k] = v

        # Modules
        self.t_emb_start = nn.Parameter(torch.randn((1, self.hp.d_hidden)) * 0.1)
        self.t_emb_end = nn.Parameter(torch.randn((1, self.hp.d_hidden)) * 0.1)
        self.encoder = EncoderFC(dim=dim, d_hidden=self.hp.d_hidden, d_out=self.hp.embedding_dim * d_latent, d_residual=self.hp.d_residual,
                                 d_cond=(self.hp.d_sideinfo if enc_si else None))
        self.decoder = DecoderFC(d_in=self.hp.embedding_dim * d_latent, d_hidden=self.hp.d_hidden, d_residual=self.hp.d_residual, d_out=dim,
                                 d_cond=(self.hp.d_sideinfo if dec_si else None))
        self.quantizer = vqvae.VectorQuantizerEMA(2**codebook_bits, self.hp.embedding_dim, decay=self.hp.decay, eps=self.hp.eps)

        if self.use_sideinfo:
            self.si_net = SINet(dim, self.hp.d_sideinfo, self.hp.d_residual, self.hp.d_hidden, self.hp.num_residual_layers, self.hp.activation)
        else:
            self.si_net = None

    def forward(self, x, t, c=None):
        t = t.float().view(-1, 1) / self.hp.max_timestep
        assert t.min() >= 0 and t.max() <= 1.0
        t_emb = self.t_emb_start * (1-t) + self.t_emb_end * t

        if self.use_sideinfo:
            c = self.si_net(c, t_emb)

        # Encoder
        z_e = self.encoder(x, t_emb, c=c)
        zs = z_e.shape
        z_e = z_e.view(len(x), self.hp.embedding_dim, self.hp.d_latent)

        # Quantize
        z_q, emb, idx = self.quantizer(z_e)
        z_q = z_q.view(len(z_q), self.hp.embedding_dim * self.hp.d_latent)

        # Decoder
        x_rec = self.decoder(z_q, t_emb, c=c)

        return x_rec, z_q, emb, z_e.view(*zs), idx

    def decode_indices(self, indices, c=None):
        raise NotImplementedError
        # if self.use_sideinfo:
        #     c = self.si_net(c)

        # assert indices.shape[1] == self.hp.ch_latent and indices.ndim == 4
        # z_q = self.quantizer.embed(indices)
        # z_q = z_q.view(z_q.shape[0], -1, z_q.shape[3], z_q.shape[4])
        # x = self.decoder(z_q, c=c)
        # return x

    def get_autoencoder_param_count(self):
        raise NotImplementedError
        # return self.encoder.get_non_si_param_count() + self.decoder.get_non_si_param_count() + get_param_count(self.z_proj) + self.quantizer.get_param_count()


def test_stuff():
    # x = torch.rand(7, DIM).float() - 0.5
    # model = VqvaeMnistGradTc(dim=DIM, codebook_bits=4, d_latent=20, enc_si=False, dec_si=False)
    # x_rec, _, _, _, _ = model(x)
    # import ipdb; ipdb.set_trace()
    # pass
    # encoder = EncoderFC(DIM, 10, 20)
    # quantizer = vqvae.VectorQuantizerEMA(2**8, 64)
    # decoder = DecoderFC(64*10, 8, 5, DIM)
    # z_e = encoder(x)
    # z_q, _, idx = quantizer(z_e)
    # x_rec = decoder(z_q.view(-1, 64 * 10))
    pass



# test_stuff()