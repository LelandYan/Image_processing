# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/19 16:22'

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("aaaaaaaa.jpg")
# img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
# cv2.namedWindow("hull", cv2.WINDOW_NORMAL)
# cv2.imshow("hull", img)
# cv2.waitKey()
# cv2.destroyAllWindows()
from scipy import ndimage as ndi

img = img[:, :, 0]
plt.figure('hist_plot')
arr = img.flatten()
plt.hist(arr, 256)
plt.show()

rows, cols = img.shape
labels = np.zeros([rows, cols])
for i in range(rows):
    for j in range(cols):
        if (img[i, j] > 70):
            labels[i, j] = 1
        else:
            labels[i, j] = 0

cv2.namedWindow("labels", cv2.WINDOW_NORMAL)
cv2.imshow("labels", labels)
cv2.waitKey(0)
cv2.destroyAllWindows()
