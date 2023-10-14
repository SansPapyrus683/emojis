from torch import nn


class Generator(nn.Module):
    def __init__(self, side_len: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=100,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )

    def forward(self, z):
        z = self.conv(z)
        return z


class Discriminator(nn.Module):
    def __init__(self, side_len: int) -> None:
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=1,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False
        )

    def forward(self, image):
        x = self.sigmoid(self.conv(image))
        return x
