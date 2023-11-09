import torch.nn as nn


class DCGANDiscriminator(nn.Module):
    def __init__(self, isize, nc, ndf, n_extra_layers=0):
        super().__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module(
            "initial:{0}-{1}:conv".format(nc, ndf),
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        )
        main.add_module("initial:{0}:relu".format(ndf), nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(
                "extra-layers-{0}:{1}:conv".format(t, cndf),
                nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False),
            )
            main.add_module(
                "extra-layers-{0}:{1}:batchnorm".format(t, cndf), nn.BatchNorm2d(cndf)
            )
            main.add_module(
                "extra-layers-{0}:{1}:relu".format(t, cndf),
                nn.LeakyReLU(0.2, inplace=True),
            )

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module(
                "pyramid:{0}-{1}:conv".format(in_feat, out_feat),
                nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False),
            )
            main.add_module(
                "pyramid:{0}:batchnorm".format(out_feat), nn.BatchNorm2d(out_feat)
            )
            main.add_module(
                "pyramid:{0}:relu".format(out_feat), nn.LeakyReLU(0.2, inplace=True)
            )
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module(
            "final:{0}-{1}:conv".format(cndf, 1),
            nn.Conv2d(cndf, 1, 4, 1, 0, bias=False),
        )
        self.main = main

    def forward(self, inp):
        output = self.main(inp)
        output = output.mean(0)
        return output.view(1)


class DCGANGenerator(nn.Module):
    def __init__(self, isize, nz, nc, ngf, n_extra_layers=0):
        super().__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module(
            "initial:{0}-{1}:convt".format(nz, cngf),
            nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False),
        )
        main.add_module("initial:{0}:batchnorm".format(cngf), nn.BatchNorm2d(cngf))
        main.add_module("initial:{0}:relu".format(cngf), nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize // 2:
            main.add_module(
                "pyramid:{0}-{1}:convt".format(cngf, cngf // 2),
                nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False),
            )
            main.add_module(
                "pyramid:{0}:batchnorm".format(cngf // 2), nn.BatchNorm2d(cngf // 2)
            )
            main.add_module("pyramid:{0}:relu".format(cngf // 2), nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(
                "extra-layers-{0}:{1}:conv".format(t, cngf),
                nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False),
            )
            main.add_module(
                "extra-layers-{0}:{1}:batchnorm".format(t, cngf), nn.BatchNorm2d(cngf)
            )
            main.add_module("extra-layers-{0}:{1}:relu".format(t, cngf), nn.ReLU(True))

        main.add_module(
            "final:{0}-{1}:convt".format(cngf, nc),
            nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False),
        )
        main.add_module("final:{0}:tanh".format(nc), nn.Tanh())
        self.main = main

    def forward(self, inp):
        return self.main(inp)


class MLPGenerator(nn.Module):
    def __init__(self, isize, nz, nc, ngf):
        super().__init__()
        main = nn.Sequential(
            # Z goes into a linear of size: ngf
            nn.Linear(nz, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, nc * isize * isize),
        )
        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz

    def forward(self, inp):
        inp = inp.view(inp.size(0), inp.size(1))
        output = self.main(inp)
        return output.view(output.size(0), self.nc, self.isize, self.isize)


class MLPDiscriminator(nn.Module):
    def __init__(self, isize, nz, nc, ndf):
        super().__init__()
        main = nn.Sequential(
            # Z goes into a linear of size: ndf
            nn.Linear(nc * isize * isize, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, 1),
        )
        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz

    def forward(self, inp):
        inp = inp.view(inp.size(0), inp.size(1) * inp.size(2) * inp.size(3))
        output = self.main(inp)
        output = output.mean(0)
        return output.view(1)
