import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint
from torch.nn.utils import spectral_norm

from .type import Callable


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        n_hidden_layers: int = 1,
        nonlinearity: Callable = nn.ReLU,
        last_nonlinearity: bool = False,
        zero_init_last: bool = False,
        apply_spectral_norm: bool = False,
    ):
        super().__init__()
        assert n_hidden_layers >= 1
        self.in_dim = in_dim
        self.out_dim = h_dim

        module_list = []

        sn: Callable = (
            spectral_norm if apply_spectral_norm else lambda x: x  # type: ignore
        )

        module_list.append(sn(nn.Linear(in_dim, h_dim)))
        module_list.append(nonlinearity())

        for _ in range(n_hidden_layers - 1):
            module_list.append(sn(nn.Linear(h_dim, h_dim)))
            module_list.append(nonlinearity())

        module_list.append(sn(nn.Linear(h_dim, out_dim)))

        if zero_init_last:
            module_list[-1].weight.data.zero_()
            module_list[-1].bias.data.zero_()
        if last_nonlinearity:
            module_list.append(nonlinearity())
        self.mlp = nn.Sequential(*module_list)

    def forward(self, x):
        return self.mlp(x)


class ODEFunc(nn.Module):
    def __init__(self, ode_net: nn.Module):
        super().__init__()
        self.ode_net = ode_net

    def forward(self, t, x):
        return self.ode_net(x)


class ODEGRU(nn.Module):
    def __init__(
        self,
        data_dim: int,
        state_dim: int,
        ode_h_dim: int,
        ode_n_layers: int,
        ode_nonlinearity: nn.Module,
        integration_step_size: float = 0.1,
        integration_method: str = "rk4",
    ):
        super().__init__()
        self.data_dim = data_dim
        self.state_dim = state_dim
        self.ode_func = ODEFunc(
            MLP(
                in_dim=state_dim,
                out_dim=state_dim,
                h_dim=ode_h_dim,
                n_hidden_layers=ode_n_layers,
                nonlinearity=ode_nonlinearity,
            )
        )
        self.integration_step_size = integration_step_size
        self.integration_method = integration_method
        self.gru_cell = nn.GRUCell(input_size=data_dim, hidden_size=state_dim)

    def gru_update(self, x, h, mask):
        h_next = self.gru_cell(x, h)
        mask = mask.unsqueeze(-1)
        h_next = mask * h_next + (1 - mask) * h
        return h_next

    def ode_update(self, h, t1, t2):
        out = odeint(
            self.ode_func,
            h,
            torch.tensor([t1, t2], device=h.device),
            method=self.integration_method,
            options={"step_size": self.integration_step_size},
        )
        return out[-1]

    def forward(self, x, times, mask):
        B, T, F = x.size()
        out = []
        if T == 1:
            h = torch.zeros(B, self.state_dim, device=x.device)
            h = self.gru_update(x[:, 0], h, mask[:, 0])
            out.append(h)
        else:
            out = []
            h = torch.zeros(B, self.state_dim, device=x.device)

            h = self.gru_update(x[:, 0], h, mask[:, 0])
            out.append(h)
            for i, (t1, t2) in enumerate(zip(times[:-1], times[1:])):
                h_ode = self.ode_update(h, t1.item(), t2.item())
                h = self.gru_update(x[:, i + 1], h_ode, mask[:, i + 1])
                out.append(h)
        out = torch.stack(out)
        out = out.permute(1, 0, 2)
        return out


class MergeLastDims(nn.Module):
    def __init__(self, ndims=1):
        super().__init__()
        self.ndims = ndims

    def forward(self, x):
        shape = x.shape
        last_dim = np.prod(shape[-self.ndims :])
        x = x.view(shape[: -self.ndims] + (last_dim,))
        return x


class ImageEncoder(nn.Module):
    def __init__(self, img_size, channels, out_dim):
        super().__init__()
        self.img_size = img_size
        self.ch = channels
        self.out_dim = out_dim
        self.network = nn.Sequential(
            nn.ZeroPad2d(padding=[0, 1, 0, 1]),
            nn.Conv2d(
                in_channels=self.ch,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            nn.ReLU(),
            nn.ZeroPad2d(padding=[0, 1, 0, 1]),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            nn.ReLU(),
            nn.ZeroPad2d(padding=[0, 1, 0, 1]),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            nn.ReLU(),
            MergeLastDims(ndims=3),
            nn.Linear(32 * 4 * 4, self.out_dim),
        )

    def forward(self, x):
        assert x.dim() == 3, "Expected 3 dims B, T, C * H * W"
        B, T, _ = x.shape
        x = x.view(B, T, self.ch, self.img_size, self.img_size)
        B, T, C, H, W = x.shape
        assert H == W
        x = x.reshape(B * T, C, H, W)
        h = self.network(x)
        h = h.view(B, T, self.out_dim)
        return h


class ImageDecoder(nn.Module):
    def __init__(self, in_dim, img_size, channels):
        super().__init__()
        self.img_size = img_size
        self.ch = channels
        self.linear = nn.Linear(
            in_features=in_dim,
            out_features=32 * 4 * 4,
        )
        self.convnet = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32 * 2**2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(
                in_channels=32,
                out_channels=32 * 2**2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(
                in_channels=32,
                out_channels=32 * 2**2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(
                in_channels=32,
                out_channels=self.ch,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x):
        assert x.dim() == 3, "Expected 3 dims B, T, D"
        B, T, _ = x.shape
        h = self.linear(x)
        h = h.view(B * T, 32, 4, 4)
        h = self.convnet(h)
        h = h.view(B, T, self.ch, self.img_size, self.img_size)
        return h


class Lambda(nn.Module):
    def __init__(self, func: Callable):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
