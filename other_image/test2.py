# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/19 7:23'

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
# image = cv2.imread("./raw_data/1.jpg")
# cv2.imwrite("canny.jpg", cv2.Canny(image, 200, 300))
# cv2.namedWindow("canny", cv2.WINDOW_NORMAL)
# cv2.imshow("canny", cv2.imread("canny.jpg"))
# cv2.waitKey()
# cv2.destroyAllWindows()

img = cv2.pyrDown(cv2.imread("./raw_data/1.jpg", cv2.IMREAD_UNCHANGED))
# threshold 函数对图像进行二化值处理，由于处理后图像对原图像有所变化，因此img.copy()生成新的图像，cv2.THRESH_BINARY是二化值
# ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
thresh = median(thresh, sm.morphology.disk(5))
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, op= cv2.MORPH_OPEN,kernel=kernel,iterations=1)
# sure_bg = cv2.dilate(opening,kernel,iterations=3)#膨胀
# cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)
# cv2.imshow("thresh", opening)
# cv2.waitKey()
# cv2.destroyAllWindows()

# findContours函数查找图像里的图形轮廓
# 函数参数thresh是图像对象
# 层次类型，参数cv2.RETR_EXTERNAL是获取最外层轮廓，cv2.RETR_TREE是获取轮廓的整体结构
# 轮廓逼近方法
# 输出的返回值，image是原图像、contours是图像的轮廓、hier是层次类型
image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    # 轮廓绘制方法一
    # boundingRect函数计算边框值，x，y是坐标值，w，h是矩形的宽和高
    # x, y, w, h = cv2.boundingRect(c)
    # # 在img图像画出矩形，(x, y), (x + w, y + h)是矩形坐标，(0, 255, 0)设置通道颜色，2是设置线条粗度
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 轮廓绘制方法二
    # 查找最小区域
    # rect = cv2.minAreaRect(c)
    # # 计算最小面积矩形的坐标
    # box = cv2.boxPoints(rect)
    # # 将坐标规范化为整数
    # box = np.int0(box)
    # # 绘制矩形
    # cv2.drawContours(img, [box], 0, (0, 0, 255), 3)

    # 轮廓绘制方法三
    # 圆心坐标和半径的计算
    (x, y), radius = cv2.minEnclosingCircle(c)
    # 规范化为整数
    center = (int(x), int(y))
    radius = int(radius)
    # 勾画圆形区域
    img = cv2.circle(img, center, radius, (0, 255, 0), 2)

# # 轮廓绘制方法四
# 围绕图形勾画蓝色线条
cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
print(len(contours))
# 显示图像
cv2.imshow("contours", img)
cv2.waitKey()
cv2.destroyAllWindows()

