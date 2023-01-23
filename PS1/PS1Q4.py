import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import color, io

class Prob4():
    def __init__(self):
        """Load input color image indoor.png and outdoor.png here as class variables."""

        self.indoor = None
        self.outdoor = None
        ###### START CODE HERE ######
        
        self.indoor = io.imread('indoor.png')
        self.outdoor = io.imread('outdoor.png')

        ###### END CODE HERE ######

    
    def prob_4_1(self):
        """Plot R,G,B channels separately and also their corresponding LAB space channels separately"""
        
        ###### START CODE HERE ######

        imgplot = plt.imshow(self.indoor[:,:,0], cmap = 'gray')
        plt.title('Indoor Red')
        plt.show()

        imgplot = plt.imshow(self.indoor[:,:,1], cmap = 'gray')
        plt.title('Indoor Green')
        plt.show()

        imgplot = plt.imshow(self.indoor[:,:,2], cmap = 'gray')
        plt.title('Indoor Blue')
        plt.show()
        
        ###### END CODE HERE ######
        return

    def prob_4_2(self):
        """
        Convert the loaded RGB image to HSV and return HSV matrix without using inbuilt functions. Return the HSV image as HSV. Make sure to use a 3 channeled RGB image with floating point values lying between 0 - 1 for the conversion to HSV.

        Returns:
            HSV image (3 channeled image with floating point values lying between 0 - 1 in each channel)
        """
        
        HSV = None
        ###### START CODE HERE ######
        
        ###### END CODE HERE ######
        return HSV

        
if __name__ == '__main__':
    
    p4 = Prob4()
    p4.prob_4_1()
    HSV = p4.prob_4_2()





