import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import color, io
from scipy.ndimage import convolve, gaussian_filter


class Prob5():
    def __init__(self):
        """Load input color images cat.bmp and dog.bmp here as class variables self.cat and self.dog.
        Convert them to contain floating point values between 0 and 1 instead of integer values between 0 and 255."""

        self.cat = None
        self.dog = None

        self.identity_filter = np.asarray([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        self.blur_filter = np.ones((3,3)) / 9.
        self.sobel_filter = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.laplacian_filter = np.asarray([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        self.gaussian_filter = self.get_gaussian_filter()

        ###### START CODE HERE ######

        ###### END CODE HERE ######

    
    def get_gaussian_filter(self, sigma=7):
        """Creates a low pass Gaussian filter given the standard deviation."""
        x = np.arange(4*sigma + 1)
        mean = sigma // 2

        xx, yy = np.meshgrid(x, x)
        filter = np.exp(-0.5*(xx**2 + yy**2) / sigma**2)
        return filter / np.sum(filter)

    def vis_image_scales_numpy(self, image):
        """
        This function will display an image at different scales (zoom factors). The
        original image will appear at the far left, and then the image will
        iteratively be shrunk by 2x in each image to the right.
        This is a particular effective way to simulate the perspective effect, as
        if viewing an image at different distances. We thus use it to visualize
        hybrid images, which represent a combination of two images, as described
        in the SIGGRAPH 2006 paper "Hybrid Images" by Oliva, Torralba, Schyns.
        Args:
            image: Array of shape (H, W, C)
        Returns:
            img_scales: Array of shape (M, K, C) representing horizontally stacked
                images, growing smaller from left to right.
                K = W + int(1/2 W + 1/4 W + 1/8 W + 1/16 W) + (5 * 4)
        """
        original_height = image.shape[0]
        original_width = image.shape[1]
        num_colors = 1 if image.ndim == 2 else 3
        img_scales = np.copy(image)
        cur_image = np.copy(image)

        scales = 5
        scale_factor = 0.5
        padding = 5

        new_h = original_height
        new_w = original_width

        for scale in range(2, scales + 1):
            # add padding
            img_scales = np.hstack(
                (
                    img_scales,
                    np.ones((original_height, padding, num_colors), dtype=np.float32),
                )
            )

            new_h = int(scale_factor * new_h)
            new_w = int(scale_factor * new_w)
            # downsample image iteratively
            cur_image = cv2.resize(cur_image, (new_w, new_h))

            # pad the top to append to the output
            h_pad = original_height - cur_image.shape[0]
            pad = np.ones((h_pad, cur_image.shape[1], num_colors), dtype=np.float32)
            tmp = np.vstack((pad, cur_image))
            img_scales = np.hstack((img_scales, tmp))

        return img_scales


    def my_conv2d(self, image, filter):
        """Apply a 2D convolution with the given filter to the image. Use scipy.ndimage.convolve with the default parameters
        to apply the convolution to each channel individually. Clip the resulting values to range [0,1].
        
        Returns:
            convolved_image: the image with the filter applied
        """
        convolved_image = None
        ###### START CODE HERE ######

        ###### END CODE HERE ######
        return convolved_image


    def prob_5_1(self):
        """Apply the identity filter, blur filter, sobel filter, and laplacian filter (given as class variables)
        to the cat image (self.cat). Return each result image. Display them in your report.
        
        Returns:
            identity_image: the cat image with the identity filter applied
            blur_image: the cat image with the blur filter applied
            sobel_image: the cat image with the sobel filter applied
            laplacian_image: the cat image with the laplacian filter applied
        """

        identity_image = None
        blur_image = None
        sobel_image = None
        laplacian_image = None
        ###### START CODE HERE ######

        ###### END CODE HERE ######
        return identity_image, blur_image, sobel_image, laplacian_image

    
    def create_hybrid_image(self, image1, image2):
        """
        Takes two images and creates a hybrid image using the low pass Gaussian filter.
        The hybrid image is created by adding the low frequency content from image1 and 
        the high frequency content of image2.
        HINTS:
        - You will use the my_conv2d() function with self.gaussian_filter to get the low 
        frequency content of each image.
        - You can obtain the high frequency content of an image by removing its low
        frequency content. Think about how to do this in mathematical terms.
        - Clip the resulting hybrid image values to range [0,1].

        Returns:
            hybrid_image: the hybrid image of image1 and image2
        """
        hybrid_image = None
        ###### START CODE HERE ######

        ###### END CODE HERE ######
        return hybrid_image


    def prob_5_2(self):
        """Call create_hybrid_image() on self.dog and self.cat and display the resulting image. This
        image combines the low frequencies from the dog image and the high frequencies from the cat image.
        Additionally, call self.vis_image_scales_numpy() on your hybrid image and display the result. This
        function generates a visualization showing the hybrid image at different scales. Return the hybrid
        image and hybrid image visualization. Include both in your report

        Returns:
            hybrid_image: the hybrid image created from the dog and cat images
            hybrid_image_vis: the multiscale visualization created from calling self.vis_image_scaled_numpy()
                on your hybrid image        
        """

        hybrid_image = None
        hybrid_image_vis = None 
        ###### START CODE HERE ######

        ###### END CODE HERE ######
        return hybrid_image, hybrid_image_vis


        
if __name__ == '__main__':

    p5 = Prob5()
    identity_image, blur_image, sobel_image, laplacian_image = p5.prob_5_1()
    hybrid_image, hybrid_image_vis = p5.prob_5_2()
