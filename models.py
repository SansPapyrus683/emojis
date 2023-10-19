from torch import nn
from torch.nn import functional as F

from utils import conv_sz, conv_t_sz


class Generator(nn.Module):
    def __init__(self,
        in_channels: int = 100,
        out_channels: int = 4,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1
    ) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

    def forward(self, z):
        z = self.conv(z)
        return z


class Discriminator(nn.Module):
    def __init__(
        self,
        side_len: int = 64,
        in_channels: int = 4,
        out_channels: int = 1,
        kernel_size: int = 4,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()

        conv_size = conv_sz(
            side_len, side_len,
            stride, padding, kernel_size, 1
        )
        flatten_size = out_channels * conv_size[0] * conv_size[1]

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.fc1 = nn.Linear(flatten_size, 1)


    def forward(self, image):
        x = F.relu(self.conv1(image))
        x = x.flatten(start_dim=1)
        x = F.sigmoid(self.fc1(x))
        return x
