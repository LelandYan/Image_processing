# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/19 7:47'

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import skimage as sm
from skimage import morphology
from skimage.feature import peak_local_max
from skimage.io import imshow
from skimage.color import rgb2gray
from skimage.filters.rank import median
from skimage.measure import find_contours

################################################################################

print('Load Image')

imgFile = './raw_data/1.jpg'

# load an original image
img = cv2.imread(imgFile)
################################################################################

# color value range
cRange = 256

rows, cols, channels = img.shape

# convert color space from bgr to gray
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
################################################################################
thresh = median(imgGray, sm.morphology.disk(5))
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.erode(thresh,kernel,iterations=3)#膨胀
# laplacian edge
imgLap = cv2.Laplacian(thresh, cv2.CV_8U)

# otsu method
threshold, imgOtsu = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# adaptive gaussian threshold
imgAdapt = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# imgAdapt = cv2.medianBlur(imgAdapt, 3)
################################################################################

canny = cv2.Canny(imgAdapt, 10, 20)
# 霍夫变换圆检测
circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 50, param1=80, param2=30, minRadius=0, maxRadius=50)
# 输出返回值，方便查看类型
# print(circles)

# # 输出检测到圆的个数
print(len(circles[0]))
#
# print('-------------我是条分割线-----------------')
# 根据检测到圆的信息，画出每一个圆
for circle in circles[0]:
    # 圆的基本信息
    print(circle[2])
    # 坐标行列(就是圆心)
    x = int(circle[0])
    y = int(circle[1])
    # 半径
    r = int(circle[2])
    # 在原图用指定颜色圈出圆，参数设定为int所以圈画存在误差
    img = cv2.circle(img, (x, y), r, (255, 255,255), 1, 8, 0)
# 显示新图像
cv2.namedWindow("binary2", cv2.WINDOW_NORMAL)
cv2.imshow('binary2', img)

# 按任意键退出
cv2.waitKey(0)
cv2.destroyAllWindows()






# display original image and gray image
# plt.subplot(2, 2, 1), plt.imshow(img), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 2), plt.imshow(imgLap, cmap='gray'), plt.title('Laplacian Edge'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 3), plt.imshow(imgOtsu, cmap='gray'), plt.title('Otsu Method'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 4), plt.imshow(imgAdapt, cmap='gray'), plt.title('Adaptive Gaussian Threshold'), plt.xticks(
#     []), plt.yticks([])
# plt.show()
# ################################################################################
#
# print('Goodbye!')