# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/15 16:21'

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("2.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

kernel = np.ones((9, 9), np.uint8)
img_open = cv2.morphologyEx(img, op= cv2.MORPH_OPEN, kernel=kernel)
img_close = cv2.morphologyEx(img, op= cv2.MORPH_CLOSE, kernel=kernel)
img_grad = cv2.morphologyEx(img, op= cv2.MORPH_GRADIENT, kernel=kernel)
img_tophat = cv2.morphologyEx(img, op= cv2.MORPH_TOPHAT, kernel=kernel)
img_blackhat = cv2.morphologyEx(img, op= cv2.MORPH_BLACKHAT, kernel=kernel)
# Plot the images
images = [img, img_open, img_close, img_grad,
          img_tophat, img_blackhat]
fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (15, 15))
for ind, p in enumerate(images):
    ax = axs[ind//3, ind%3]
    ax.imshow(p, cmap = 'gray')
    ax.axis('off')
plt.show()
# img_grad = cv2.cvtColor(img_grad,cv2.COLOR_BAYER_BG2BGR)
# cv2.imwrite("gradient_image.jpg",img_grad)


# gray = cv2.cvtColor(img_tophat,cv2.COLOR_BAYER_GR2GRAY)
# ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#
# contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
# cv2.imshow("img",img)
# plt.show()