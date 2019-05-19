# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/18 22:04'

import matplotlib.pyplot as plt
import cv2

img = cv2.imread("gradient_image1.jpg")
img = img[:,:,0]
plt.figure('hist_plot')
arr = img.flatten()
plt.hist(arr,256)
plt.show()