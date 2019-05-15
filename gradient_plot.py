# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/15 17:02'

import cv2
import matplotlib.pyplot as plt
img = cv2.imread("gradient_image.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (9, 9),0)
ret, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
closed = cv2.dilate(binary, None, iterations=110)
closed = cv2.erode(closed, None, iterations=120)

_,contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

cv2.namedWindow('demo', 0)
cv2.imshow("demo", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

#
#
#

# 傅里叶变化
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# import copy
# img = cv2.imread('gradient_image.jpg',0)
# f = np.fft.fft2(img)
# fshift = np.fft.fftshift(f)
#
# rows,cols = img.shape
# crow,ccol = int(rows/2) , int(cols/2)
# for i in range(crow-30,crow+30):
#     for j in range(ccol-30,ccol+30):
#         fshift[i][j]=0.0
# f_ishift = np.fft.ifftshift(fshift)
# img_back = np.fft.ifft2(f_ishift)#进行高通滤波
# # 取绝对值
# img_back = np.abs(img_back)
# plt.subplot(121),plt.imshow(img,cmap = 'gray')#因图像格式问题，暂已灰度输出
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# #先对灰度图像进行伽马变换，以提升暗部细节
# rows,cols = img_back.shape
# gamma=copy.deepcopy(img_back)
# rows=img.shape[0]
# cols=img.shape[1]
# for i in range(rows):
#     for j in range(cols):
#         gamma[i][j]=5.0*pow(gamma[i][j],0.34)#0.34这个参数是我手动调出来的，根据不同的图片，可以选择不同的数值
# #对灰度图像进行反转
#
# for i in range(rows):
#     for j in range(cols):
#         gamma[i][j]=255-gamma[i][j]
# plt.subplot(122),plt.imshow(gamma,cmap = 'gray')
# plt.title('Result in HPF'), plt.xticks([]), plt.yticks([])
# plt.show()

# Canny边缘检测
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread('gradient_image.jpg',0)
# edges = cv2.Canny(img,100,200)
#
# plt.subplot(121),plt.imshow(img,cmap='gray')
# plt.title('original'),plt.xticks([]),plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap='gray')
# plt.title('edge'),plt.xticks([]),plt.yticks([])
#
# plt.show()


