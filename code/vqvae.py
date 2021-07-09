"""
VQ-VAE implementation based on:
    https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    def __init__(self, codebook_size, embedding_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps

        self.register_buffer('emb', torch.randn([embedding_dim, codebook_size]) * 0.05)
        self.register_buffer('emb_ema', self.emb.clone())
        self.register_buffer('counts_ema', torch.zeros(codebook_size).type_as(self.emb))

    def forward(self, z_e):
        # Assume dim=1 corresponds to the encodings
        assert z_e.shape[1] == self.embedding_dim
        orig_rank = z_e.ndim
        batch_shape = z_e.shape[2:]
        # (N, embedding_dim, *) -> (N, *, embedding_dim)
        z_e = z_e.permute(0, *range(2, orig_rank), 1).reshape(-1, self.embedding_dim)
        dist = (z_e ** 2).sum(1, keepdim=True) - 2 * torch.mm(z_e, self.emb) + (self.emb ** 2).sum(0, keepdim=True)
        assert dist.shape == (z_e.shape[0], self.codebook_size)

        idx = dist.argmin(1)
        embeddings = F.embedding(idx, self.emb.t())
        z_q = z_e + (embeddings - z_e).detach()
        assert embeddings.shape == z_e.shape == z_q.shape

        if self.training:
            with torch.no_grad():
                idx_onehot = F.one_hot(idx, self.codebook_size).type_as(self.emb)

                # EMA update for counts
                self.counts_ema.data.add_((1 - self.decay) * (idx_onehot.sum(0) - self.counts_ema))
                # self.counts_ema =  self.decay_rate * self.counts_ema + (1-self.decay_rate) * idx_onehot.sum(0)

                # EMA update for embedding (codebook)
                emb_update = z_e.t() @ idx_onehot
                assert emb_update.shape == (self.embedding_dim, self.codebook_size)
                self.emb_ema.data.add_((1 - self.decay) * (emb_update - self.emb_ema))
                # self.emb_ema =  self.decay_rate * self.emb_ema + (1-self.decay_rate) * emb_update
                n = self.counts_ema.sum()
                counts = (self.counts_ema + self.eps) / (n + self.codebook_size * self.eps) * n
                normalized_emb = self.emb_ema / counts[None]
                assert normalized_emb.shape == self.emb.shape
                self.emb.data.copy_(normalized_emb)

        # Reshape to original shape
        z_q = z_q.view(-1, *batch_shape, self.embedding_dim) \
                .permute(0, -1, *range(1,orig_rank-1)).contiguous()
        idx = idx.view(-1, *batch_shape)

        return z_q, None, idx

    def embed(self, idx):
        embeddings = F.embedding(idx, self.emb.t())
        embeddings = embeddings.permute(0, -1, *range(1, idx.ndim)).contiguous()
        return embeddings

    def get_param_count(self):
        return self.embedding_dim * self.codebook_size + self.codebook_size


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        # In Sonnet, the following initialization uses truncated normal
        fan_in = np.prod(self.weight.shape[1:])
        self.weight.data.normal_(mean=0., std=1. / np.sqrt(fan_in))
        self.bias.data.zero_()


class ConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        # In Sonnet, the following initialization uses truncated normal
        fan_in = np.prod(self.weight.shape[2:]) * in_channels
        self.weight.data.normal_(mean=0., std=1. / np.sqrt(fan_in))
        self.bias.data.zero_()


class ResidualBlock(nn.Module):
    def __init__(self, ch_hidden: int, ch_residual: int, num_residual_layers: int, activation: str):
        super().__init__()
        self.layers = nn.ModuleList()
        act_class = {
            'relu': nn.ReLU,
            'elu': nn.ELU,
            'gelu': nn.GELU,
        }[activation]
        for _ in range(num_residual_layers):
            self.layers.append(nn.Sequential(
                act_class(),
                Conv2d(ch_hidden, ch_residual, kernel_size=3, stride=1, padding=1),
                act_class(),
                Conv2d(ch_residual, ch_hidden, kernel_size=1, stride=1, padding=0),
            ))

    def forward(self, h):
        for layer in self.layers:
            h = h + layer(h)
        return h

