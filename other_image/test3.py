# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/19 7:42'

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
# dst = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
# img = cv2.pyrDown(dst, cv2.IMREAD_UNCHANGED)
# # ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) , 127, 255, cv2.THRESH_BINARY)
# thresh = cv2.adaptiveThreshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
# thresh = median(thresh, sm.morphology.disk(5))
# cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)
# cv2.imshow("thresh", thresh)
# cv2.waitKey()
# cv2.destroyAllWindows()
###################################################################

##################################################################
# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)

# threshold, imgOtsu = cv2.threshold(thresh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cv2.namedWindow("hull", cv2.WINDOW_NORMAL)
# cv2.imshow("hull", imgOtsu)
# cv2.waitKey()
# cv2.destroyAllWindows()
# cv2.namedWindow("hull", cv2.WINDOW_NORMAL)
# cv2.imshow("hull", thresh)
# cv2.waitKey()
# cv2.destroyAllWindows()
# findContours函数查找图像里的图形轮廓
# 函数参数thresh是图像对象
# 层次类型，参数cv2.RETR_EXTERNAL是获取最外层轮廓，cv2.RETR_TREE是获取轮廓的整体结构
# 轮廓逼近方法
# 输出的返回值，image是原图像、contours是图像的轮廓、hier是层次类型
#########################################################################################
image = cv2.imread("./raw_data/4.jpg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)


# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)
opening = cv2.bilateralFilter(opening,9,80,80)
opening = median(opening, sm.morphology.disk(3))
# opening = cv2.morphologyEx(opening,cv2.MORPH_GRADIENT,kernel, iterations = 1)
######################################################################################
th, im_th = cv2.threshold(opening, 220, 255, cv2.THRESH_BINARY_INV)
# sure_bg = cv2.dilate(opening,kernel,iterations=2)
# Copy the thresholded image.
im_floodfill = im_th.copy()

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = im_th.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0, 0), 255)

opening = im_floodfill
opening = cv2.erode(opening,kernel,iterations=7)
#########################################################################################
image, contours, hier = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 创建新的图像black
black = cv2.cvtColor(np.zeros((image.shape[1], image.shape[0]), dtype=np.uint8), cv2.COLOR_GRAY2BGR)

counter = 0
for p,cnt in enumerate(contours):

    area = cv2.contourArea(contours[p])
    if area < 30:
        print("$$$$")
        continue
    # 轮廓周长也被称为弧长。可以使用函数 cv2.arcLength() 计算得到。这个函数的第二参数可以用来指定对象的形状是闭合的（True） ，还是打开的（一条曲线）
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    # 函数approxPolyDP来对指定的点集进行逼近，cnt是图像轮廓，epsilon表示的是精度，越小精度越高，因为表示的意思是是原始曲线与近似曲线之间的最大距离。
    # 第三个函数参数若为true,则说明近似曲线是闭合的，它的首位都是相连，反之，若为false，则断开。
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    # convexHull检查一个曲线的凸性缺陷并进行修正，参数cnt是图像轮廓。
    hull = cv2.convexHull(cnt)
    # 勾画图像原始的轮廓
    cv2.drawContours(black, [cnt], -1, (0, 255, 0), 2)
    # 用多边形勾画轮廓区域
    cv2.drawContours(black, [approx], -1, (255, 255, 0), 2)
    # 修正凸性缺陷的轮廓区域
    cv2.drawContours(black, [hull], -1, (0, 0, 255), 2)
    counter+=1
# 显示图像
print(counter)
plt.imshow(black)

cv2.namedWindow("hull", cv2.WINDOW_NORMAL)
cv2.imshow("hull", black)
cv2.waitKey()
cv2.destroyAllWindows()
from scipy import ndimage as ndi
# labels = dst
distance = ndi.distance_transform_edt(opening) #距离变换
# min_distance：最小的像素在2×min_distance + 1区分离（即峰峰数至少min_distance分隔）。找到峰值的最大数量，使用min_distance = 1。
# exclude_border：不排除峰值在图像的边界
# indices：False会返回和数组相同大小的布尔数组，为True时，会返回峰值的坐标
local_maxi =peak_local_max(distance, exclude_border = 0,min_distance = 12,indices=False,
                                   footprint=np.ones((10, 10)),labels=opening) #寻找峰值
markers = ndi.label(local_maxi)[0] #初始标记点
label_ =morphology.watershed(-distance, markers, mask=opening) #基于距离变换的分水岭算法

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
axes = axes.ravel()
ax0, ax1, ax2, ax3 = axes

ax0.imshow(opening, cmap=plt.cm.gray)#, interpolation='nearest')
ax0.set_title("Original")
ax1.imshow(-distance, cmap=plt.cm.jet, interpolation='nearest')
ax1.set_title("Distance")
ax2.imshow(sm.morphology.dilation(markers,sm.morphology.square(10)), cmap=plt.cm.Spectral, interpolation='nearest')
ax2.set_title("Markers")
ax3.imshow(label_, cmap=plt.cm.Spectral, interpolation='nearest')
ax3.set_title("Segmented")
for ax in axes:
    ax.axis('off')

fig.tight_layout()
plt.show()
