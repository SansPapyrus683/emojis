from torch import nn
from torch.nn import functional as F

SIDE = 64


class Generator64(nn.Module):
    def __init__(self,
        in_channels: int = 100,
        out_channels: int = 3
    ) -> None:
        super().__init__()

        mid = 64
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels, mid * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(mid * 8),
            nn.ReLU(True),
            # state size. (mid*8) x 4 x 4
            nn.ConvTranspose2d(mid * 8, mid * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(mid * 4),
            nn.ReLU(True),
            # state size. (mid*4) x 8 x 8
            nn.ConvTranspose2d(mid * 4, mid * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(mid * 2),
            nn.ReLU(True),
            # state size. (mid*2) x 16 x 16
            nn.ConvTranspose2d(mid * 2, mid, 4, 2, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(True),
            # state size. (mid) x 32 x 32
            nn.ConvTranspose2d(mid, out_channels, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def forward(self, z):
        return self.main(z)


class Discrim64(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()

        mid = 64
        self.main = nn.Sequential(
            # input is (in_channels) x 64 x 64
            nn.Conv2d(in_channels, mid, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (mid) x 32 x 32
            nn.Conv2d(mid, mid * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(mid * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (mid*2) x 16 x 16
            nn.Conv2d(mid * 2, mid * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(mid * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (mid*4) x 8 x 8
            nn.Conv2d(mid * 4, mid * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(mid * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (mid*8) x 4 x 4
            nn.Conv2d(mid * 8, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Sigmoid()
        )


    def forward(self, image):
        return self.main(image)
