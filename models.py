from torch import nn
from torch.nn import functional as F

from utils import conv_sz, conv_t_sz

class Generator(nn.Module):
    def __init__(self,
        in_channels: int = 100,
        out_channels: int = 4,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        output_size: int = 64,
    ) -> None:
        super().__init__()
        self.convT = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels, out_channels * 8, kernel_size, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(out_channels * 8, out_channels * 4, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( out_channels * 4, out_channels * 2, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( out_channels * 2, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, z):
        z = self.convT(z)
        return z


class Discriminator(nn.Module):
    def __init__(
        self,
        fl_channels: int = 64,
        in_channels: int = 4,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            # input is ``64 x 64 x 4``
            nn.Conv2d(in_channels, fl_channels, kernel_size, stride, padding, bias=False),
            nn.ReLU(True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(fl_channels, fl_channels * 2, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(fl_channels * 2),
            nn.ReLU(True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(fl_channels * 2, fl_channels * 4, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(fl_channels * 4),
            nn.ReLU(True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(fl_channels * 4, fl_channels * 8, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(fl_channels * 8),
            nn.ReLU(True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(fl_channels * 8, 1, kernel_size, 1, 0, bias=False),
            # state size. 1 x 1 x 1
            nn.Sigmoid()
        )


    def forward(self, image):
        z = self.conv(image)
        return z
