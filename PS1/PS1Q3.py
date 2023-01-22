import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io


class Prob3():
    def __init__(self):
        """Load input color image inputPS1Q3.jpg here and assign it as a class variable"""
        self.img = None
        ###### START CODE HERE ######
        self.img = io.imread('inputPS1Q3.jpg')
        ###### END CODE HERE ######
    
    def rgb2gray(self, rgb):
        """
        Do RGB to Gray image conversion here. Input is the RGB image and you must return the grayscale image as gray

        Returns:
            gray: grayscale image
        """
        gray = None
        ###### START CODE HERE ######
        gray = np.dot(rgb[...,:3], [0.2989, .5870, .1140])
        ###### END CODE HERE ######
        return gray
        
    def prob_3_1(self):
        """
        Swap red and green color channels here, and return swapImg

        Returns:
            swapImg: RGB image with R and G channels swapped (3 channeled image with integer values lying between 0 - 255)
        """

        swapImg = None
        ###### START CODE HERE ######
        # print(self.img.shape)
        # print(self.img[0][0])
        swapImg = np.array([[np.array([item[1], item[0], item[2]]) for item in line] for line in self.img])
        # imgplot = plt.imshow(swapImg)
        # imgplot = plt.imshow(swapImg)
        # plt.show()

        # print(swapImg[0][0])
        # print(swapImg.shape)
        ###### END CODE HERE ######
        return swapImg

    
    def prob_3_2(self):
        """
        This function would simply call your rgb2gray function and return the grayscale image.

        Returns:
            grayImg: grayscale image (1 channeled image with integer values lying between 0 - 255)
        """
        grayImg = None
        ###### START CODE HERE ######
        grayImg = self.rgb2gray(self.img)
        # imgplot = plt.imshow(grayImg, cmap = 'gray')
        # plt.show()
        ###### END CODE HERE ######
        return grayImg
    
    def prob_3_3(self):
        """
        Convert grayscale image to its negative.

        Returns:
            negativeImg: negative image (1 channeled image with integer values lying between 0 - 255)
        """
        negativeImg = None
        ###### START CODE HERE ######
        grayImg = self.prob_3_2()
        negativeImg = np.abs(grayImg-255)
        # imgplot = plt.imshow(negativeImg, cmap = 'gray')
        # plt.show()
        ###### END CODE HERE ######
        return negativeImg
    
    def prob_3_4(self):
        """
        Create mirror image of gray scale image here.
        
        Returns:
            mirrorImg: mirror image (1 channeled image with integer values lying between 0 - 255)
        """
        mirrorImg = None
        ###### START CODE HERE ######
        grayImg = self.rgb2gray(self.img)
        mirrorImg = np.flip(grayImg, axis=1)
        # imgplot = plt.imshow(mirrorImg, cmap = 'gray')
        # plt.show()
        ###### END CODE HERE ######
        return mirrorImg
    
    def prob_3_5(self):
        """
        Average grayscale image with mirror image here.
        
        Returns:
            avgImg: average of grayscale and mirror image (1 channeled image with integer values lying between 0 - 255)
        """
        avgImg = None
        ###### START CODE HERE ######
        grayImg = self.prob_3_2()
        mirrorImg = self.prob_3_4()
        avgImg = (grayImg.astype(float) + mirrorImg.astype(float))/2
        avgImg = avgImg.astype(int)
        # imgplot = plt.imshow(avgImg, cmap = 'gray')
        # plt.show()
        ###### END CODE HERE ######
        return avgImg
    
    def prob_3_6(self):
        """
        Create noise matrix N and save as noise.npy. Add N to grayscale image, clip to ensure that max value is 255.
        
        Returns:
            noisyImg, noise: grayscale image after adding noise (1 channeled image with integer values lying between 0 - 255)
            and the noise
        """
        noisyImg, noise = [None]*2
        ###### START CODE HERE ######
        grayImg = self.prob_3_2()
        noise = (np.random.rand(900,600) * 255).astype(int)
        with open('noise.npy', 'wb') as f:
            np.save(f, noise)
        noisyImg = grayImg.astype(float) + noise
        noisyImg = np.clip(noisyImg, 0, 255)
        noisyImg = noisyImg.astype(int)

        imgplot = plt.imshow(noisyImg, cmap = 'gray')
        plt.show()

        ###### END CODE HERE ######
        return noisyImg, noise
        
        
if __name__ == '__main__': 
    
    p3 = Prob3()

    # swapImg = p3.prob_3_1()
    grayImg = p3.prob_3_2()
    negativeImg = p3.prob_3_3()
    mirrorImg = p3.prob_3_4()
    avgImg = p3.prob_3_5()
    noisyImg,_ = p3.prob_3_6()

    




