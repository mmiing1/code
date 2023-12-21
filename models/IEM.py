import math
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from PIL import Image
from torch.cuda import amp

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh
from utils.plots import color_list, plot_one_box
from utils.torch_utils import time_synchronized

import torch
import torch.nn as nn
import torch.nn.functional as F

class IEM(nn.Module):
    def __init__(self, in_channels):
        super(IEM, self).__init__()
        self.channels=in_channels
        # Initialize the Gaussian and PreWitt filters within the class constructor
        self.gaussian_filters = nn.ModuleList([self.create_gaussian_kernel(in_channels) for _ in range(4)])
        self.prewitt_x, self.prewitt_y = self.create_prewitt_operator(in_channels)
        self.detail_enhancements = nn.ModuleList([DetailEnhancementModule(in_channels) for _ in range(4)])
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

        def forward(self, x):
            L_layers = [x]
            G_layers = []

            # Apply Gaussian filters to get L layers
            for gauss_filter in self.gauss_filters:
                L = self.relu(self.norm(gauss_filter(L_layers[-1])))
                L_layers.append(L)

            # Apply detail enhancement on each L layer to get G layers
            for i in range(1, len(L_layers)):
                G = self.relu(self.norm(self.detail_enhancements[i - 1](L_layers[i])))
                G_layers.append(G)

            # Upsample and combine G layers
            for i in reversed(range(1, len(G_layers))):
                G_layers[i - 1] = self.upsample(G_layers[i]) + L_layers[i]

            return G_layers[0]

    # Additional classes needed for IEM

    @staticmethod
    def create_gaussian_kernel(in_channels, kernel_size=5, sigma=1):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = (1. / (2. * np.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                              (2 * variance)
                          )
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(in_channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                    kernel_size=kernel_size, groups=in_channels, padding=kernel_size // 2, bias=False)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False

        return gaussian_filter

    @staticmethod
    def create_prewitt_operator(in_channels):
        prewitt_x = torch.tensor([[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        prewitt_y = torch.tensor([[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]]).view(1, 1, 3, 3)

        prewitt_x = prewitt_x.repeat(in_channels, 1, 1, 1)
        prewitt_y = prewitt_y.repeat(in_channels, 1, 1, 1)

        prewitt_x_filter = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, groups=in_channels,
                                     padding=1, bias=False)
        prewitt_y_filter = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, groups=in_channels,
                                     padding=1, bias=False)

        prewitt_x_filter.weight.data = prewitt_x
        prewitt_y_filter.weight.data = prewitt_y
        prewitt_x_filter.weight.requires_grad = False
        prewitt_y_filter.weight.requires_grad = False

        return prewitt_x_filter, prewitt_y_filter

# DetailEnhancementModule class needs to be adjusted to use self.prewitt_x and self.prewitt_y
class DetailEnhancementModule(nn.Module):
    def __init__(self, in_channels):
        super(DetailEnhancementModule, self).__init__()
        # Use the PreWitt filters from the IEM class
        self.prewitt_x, self.prewitt_y = IEM.create_prewitt_operator(in_channels)
        # PreWitt operator kernels would be initialized here
        self.prewitt_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.prewitt_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        conv_out = self.conv3x3(x)
        prewitt_x_out = self.prewitt_x(x)
        prewitt_y_out = self.prewitt_y(x)
        combined = conv_out + prewitt_x_out + prewitt_y_out
        return self.conv1x1(combined)

    # Example usage
        in_channels = 3  # Assuming RGB images


