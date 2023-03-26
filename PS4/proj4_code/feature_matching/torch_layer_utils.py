#useful util functions implemented with pytorch

import torch
from torch import nn
import numpy as np
"""
Image gradients are needed for both SIFT and the Harris Corner Detector, so we
implement the necessary code only once, here.
"""


class ImageGradientsLayer(torch.nn.Module):
    """
    ImageGradientsLayer: Compute image gradients Ix & Iy. This can be
    approximated by convolving with Sobel filter.
    """
    def __init__(self):
        super(ImageGradientsLayer, self).__init__()

        # Create convolutional layer
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3,
            bias=False, padding=(1,1), padding_mode='zeros')

        # Instead of learning weight parameters, here we set the filter to be
        # Sobel filter
        self.conv2d.weight = get_sobel_xy_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of ImageGradientsLayer. We'll test with a
        single-channel image, and 1 image at a time (batch size = 1).

        Args:
        -   x: input tensor of size (num_image, channel, height, width)

        Returns:
        -   output: output of HarrisNet network, (num_image, 2, height, width)
            tensor for Ix and Iy, respectively.
        """
        return self.conv2d(x)


def get_sobel_xy_parameters() -> torch.nn.Parameter:
    """
    Populate the conv layer weights for the Sobel layer (image gradient
    approximation).

    There should be two sets of filters: each should have size (1 x 3 x 3)
    for 1 channel, 3 pixels in height, 3 pixels in width. When combined along
    the batch dimension, this conv layer should have size (2 x 1 x 3 x 3), with
    the Sobel_x filter first, and the Sobel_y filter second.

    Args:
    -   None
    Returns:
    -   Torch parameter representing (2, 1, 3, 3) conv filters
    """

    sobel_x_kernel = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]).astype(np.float32)
    sobel_y_kernel = np.array(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]).astype(np.float32)
    filters = np.concatenate(
        [
            sobel_x_kernel.reshape(1,1,3,3),
            sobel_y_kernel.reshape(1,1,3,3)
        ], axis=0)
    weight_param = torch.nn.Parameter(torch.from_numpy(filters))

    return weight_param
