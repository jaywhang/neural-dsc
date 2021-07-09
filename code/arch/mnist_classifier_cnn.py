import torch
import torch.nn as nn

from code.utils import get_param_count

class MnistClassifierCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.nc = nc = 3
        self.net = nn.Sequential(
            nn.Conv2d(1,  nc, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(nc, nc, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(nc, nc, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(nc, nc, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.fc_out = nn.Linear(4*nc, 10)

    def forward(self, x):
        assert x.ndim == 4 and x.shape[1:] == (1, 28, 28)
        y = self.net(x)
        y = y.view(len(x), 4*self.nc)
        y = self.fc_out(y)
        return y