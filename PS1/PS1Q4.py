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

        # imgplot = plt.imshow(self.outdoor[:,:,0], cmap = 'gray')
        # plt.title('outdoor Red')
        # plt.show()

        # imgplot = plt.imshow(self.outdoor[:,:,1], cmap = 'gray')
        # plt.title('outdoor Green')
        # plt.show()

        # imgplot = plt.imshow(self.outdoor[:,:,2], cmap = 'gray')
        # plt.title('outdoor Blue')
        # plt.show()
        
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
        
        image = io.imread('inputPS1Q4.jpg')
        print(image.shape)

        x, y, z = image.shape
        HSV = np.zeros(image.shape)

        imgplot = plt.imshow(image)
        # plt.title('outdoor Green')
        plt.show()

        for i in range (0, x):
            for j in range (0, y):
                rgb = image[i,j]/255
                R,G,B = rgb
                V = np.max(rgb)
                m = np.min(rgb)
                C = V - m
                S = 0
                if (V != 0):
                    S = float(C)/float(V)

                H_ = 0
                if (C != 0):
                    # print('C is', C)
                    if (V == R):
                        H_ = float(float(G)-B)/float(C)
                    elif (V == G):
                        H_ = float(float(B)-R)/float(C) + 2
                    elif (V == B):
                        H_ = float(float(R)-G)/float(C) + 4
                
                H = 0
                if (H_ < 0):
                    H = float(H_)/6 + 1
                else:
                    H = float(H_)/6
                
                HSV[i,j] = np.array([H,S,V])

        ###### END CODE HERE ######
        return HSV

        
if __name__ == '__main__':
    
    p4 = Prob4()
    p4.prob_4_1()
    HSV = p4.prob_4_2()





