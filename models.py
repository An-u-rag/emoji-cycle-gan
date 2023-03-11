import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_layer(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    layers.append(conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def deconv_layer(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels,
                  kernel_size, stride, padding, bias=False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv = conv_layer(in_channels=channels, out_channels=channels,
                               kernel_size=3, stride=1, padding=1, batch_norm=True)

    def forward(self, x):
        return x + self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, opts):
        super(Discriminator, self).__init__()
        self.opts = opts

        image_channels = 3 if self.opts.format == "RGB" else 4

        ##### TODO: Define the discriminator network#####
        self.layers = nn.Sequential(
            conv_layer(in_channels=image_channels,
                       out_channels=32, kernel_size=4),
            nn.LeakyReLU(),
            conv_layer(in_channels=32, out_channels=64, kernel_size=4),
            nn.LeakyReLU(),
            conv_layer(in_channels=64, out_channels=128, kernel_size=4),
            nn.LeakyReLU(),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )
        if self.opts.d_sigmoid:
            self.layers.append(nn.Sigmoid())
        ################################################

    def forward(self, x):
        ##### TODO: Define the forward pass#####
        out = self.layers(x)
        return out
        #######################################


class NoLeakDiscriminator(nn.Module):
    def __init__(self, opts):
        super(NoLeakDiscriminator, self).__init__()
        self.opts = opts

        image_channels = 3 if self.opts.format == "RGB" else 4

        ##### TODO: Define the discriminator network#####
        self.layers = nn.Sequential(
            conv_layer(in_channels=image_channels,
                       out_channels=32, kernel_size=4),
            nn.ReLU(),
            conv_layer(in_channels=32, out_channels=64, kernel_size=4),
            nn.ReLU(),
            conv_layer(in_channels=64, out_channels=128, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )

        if self.opts.d_sigmoid:
            self.layers.append(nn.Sigmoid())
        ################################################

    def forward(self, x):
        ##### TODO: Define the forward pass#####
        out = self.layers(x)
        return out
        #######################################


class Generator(nn.Module):
    def __init__(self, opts):
        super(Generator, self).__init__()
        self.opts = opts

        image_channels = 3 if self.opts.format == "RGB" else 4

        ##### TODO: Define the generator network######
        self.layers = nn.Sequential(
            deconv_layer(in_channels=100, out_channels=128,
                         kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            deconv_layer(in_channels=128, out_channels=64, kernel_size=4),
            nn.ReLU(),
            deconv_layer(in_channels=64, out_channels=32, kernel_size=4),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        #############################################

    def forward(self, x):
        ##### TODO: Define the forward pass#####
        out = self.layers(x)
        return out
        #######################################


class CycleGenerator(nn.Module):
    def __init__(self, opts):
        super(CycleGenerator, self).__init__()
        self.opts = opts
        image_channels = 3 if self.opts.format == "RGB" else 4

        ##### TODO: Define the cyclegan generator network######
        self.layers = nn.Sequential(
            conv_layer(in_channels=image_channels,
                       out_channels=32, kernel_size=4),
            nn.ReLU(),
            conv_layer(in_channels=32, out_channels=64, kernel_size=4),
            nn.ReLU()
        )
        self.resnet = ResNetBlock(64)

        self.glayers = nn.Sequential(
            deconv_layer(in_channels=64, out_channels=32, kernel_size=4),
            nn.ReLU(),
            deconv_layer(in_channels=32,
                         out_channels=image_channels, kernel_size=4),
            nn.Tanh()
        )
        ######################################################

    def forward(self, x):
        ##### TODO: Define the forward pass#####
        out = self.layers(x)
        out = self.resnet(out)
        out = self.glayers(out)
        return out
        #######################################


class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.mseloss = nn.MSELoss()

    def forward(self, pred, label):
        loss = self.mseloss(pred, label)
        return loss


class CycleLoss(nn.Module):
    def __init__(self):
        super(CycleLoss, self).__init__()
        self.l1loss = nn.L1Loss()

    def forward(self, pred, label):
        loss = self.l1loss(pred, label)
        return loss
