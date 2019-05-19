# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/18 19:40'

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
image = cv2.imread("./raw_data/2.jpg")
# img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# plt.imshow(img_hsv,plt.cm.hsv)
# plt.show()
# img = img_hsv[:,:,0]
# # plt.figure('hist')
# arr = img.flatten()
# plt.hist(arr,256)
# plt.show()
#
# # loop_num = 5
# # plt.imshow(img)
# # pos = plt.ginput(loop_num)
# # for i in range(loop_num):
# #     x,y = int(pos[i][1]),int(pos[i][0])
# #     print('第%d个点击的 x,y：' % int(i+1) ,'(', x , y,')')
# #     print('对应的H值为：',img[x,y],'\n')
# ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)#TRIANGLE法,，全局自适应阈值, 参数0可改为任意数字但不起作用，适用于单个波峰
# print("阈值：%s" % ret)
#
# rows,cols=img.shape
# labels=np.zeros([rows,cols])
# for i in range(rows):
#     for j in range(cols):
#         if(img[i,j]<2.0):# 0.53 0.65
#             labels[i,j]=0
#         else:
#             labels[i,j]=1
#
# shuffle =  rgb2gray(labels)
# # cv2.namedWindow('im_floodfill', 0)
# # cv2.imshow("im_floodfill", shuffle)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
#
# shuffle_ = median(shuffle, sm.morphology.disk(5))
# new_img  = rgb2gray(shuffle_)
# # cv2.namedWindow('im_floodfill', 0)
# # cv2.imshow("im_floodfill", new_img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# # cv2.namedWindow('im_floodfill', 0)
# # cv2.imshow("im_floodfill", image)
# #
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# new_img_fat = morphology.erosion(new_img,sm.morphology.square(5))
#
# plt.figure('filters',figsize=(20,20))
#
# plt.subplot(131)
# plt.title('origin image')
# imshow(new_img)
#
# plt.subplot(132)
# plt.title('fat image')
# imshow(new_img_fat)
#
# plt.subplot(133)
# plt.title('fat - orgin')
# imshow(new_img_fat - new_img)
# plt.show()
# kernel_sharpen_1 = np.array([
#         [-1,-1,-1],
#         [-1,9,-1],
#         [-1,-1,-1]])
# kernel_sharpen_2 = np.array([
#         [1,1,1],
#         [1,-7,1],
#         [1,1,1]])
# kernel_sharpen_3 = np.array([
#         [-1,-1,-1,-1,-1],
#         [-1,2,2,2,-1],
#         [-1,2,8,2,-1],
#         [-1,2,2,2,-1],
#         [-1,-1,-1,-1,-1]])/8.0
#
#
# # bitwise_and 交运算
# output_1 = cv2.filter2D(image,-1,kernel_sharpen_1)
# output_2 = cv2.filter2D(image,-1,kernel_sharpen_2)
# output_3 = cv2.filter2D(image,-1,kernel_sharpen_3)
# gray = cv2.cvtColor(output_3, cv2.COLOR_RGB2GRAY)  #把输入图像灰度化
# 直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
# ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
# ret, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# binary = median(binary, sm.morphology.disk(5))
# binary = cv2.GaussianBlur(binary,(5,5),0)
# binary= sm.morphology.dilation(binary,sm.morphology.square(3))
# binary = sm.morphology.opening(binary,sm.morphology.disk(5))
# binary = cv2.erode(binary, None, iterations=5) # 1 # 2
# binary = cv2.dilate(binary, None, iterations=3) # 1 # 2
# binary = cv2.morphologyEx(binary, op= cv2.MORPH_CLOSE,kernel=kernel)
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
thresh = median(thresh, sm.morphology.disk(5))
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, op= cv2.MORPH_OPEN,kernel=kernel,iterations=1)
sure_bg = cv2.dilate(opening,kernel,iterations=3)#膨胀
# binary = cv2.morphologyEx(binary, op= cv2.MORPH_GRADIENT,kernel=kernel)
dist_transform = cv2.distanceTransform(opening,1,5)
rets, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)#参数改小了，出现不确定区域
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)#减去前景

# cv2.namedWindow("binary2", cv2.WINDOW_NORMAL)
# cv2.imshow('binary2',unknown)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##############################################################################3
distance = ndi.distance_transform_edt(binary) #距离变换
# min_distance：最小的像素在2×min_distance + 1区分离（即峰峰数至少min_distance分隔）。找到峰值的最大数量，使用min_distance = 1。
# exclude_border：不排除峰值在图像的边界
# indices：False会返回和数组相同大小的布尔数组，为True时，会返回峰值的坐标
local_maxi =peak_local_max(distance, exclude_border = 0,min_distance = 12,indices=False,
                                   footprint=np.ones((10, 10)),labels=binary) #寻找峰值
markers = ndi.label(local_maxi)[0] #初始标记点
label_ =morphology.watershed(-distance, markers, mask=binary) #基于距离变换的分水岭算法
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
axes = axes.ravel()
ax0, ax1, ax2, ax3 = axes

ax0.imshow(binary, cmap=plt.cm.gray)#, interpolation='nearest')
ax0.set_title("Original")
ax1.imshow(-distance, cmap=plt.cm.jet, interpolation='nearest')
ax1.set_title("Distance")
ax2.imshow(sm.morphology.dilation(markers,sm.morphology.square(5)), cmap= plt.cm.Spectral, interpolation='nearest')
ax2.set_title("Markers")
ax3.imshow(label_, cmap= plt.cm.Spectral, interpolation='nearest')
ax3.set_title("Segmented")

for ax in axes:
    ax.axis('off')

fig.tight_layout()
plt.show()
#################################################################################

###########################################################################
contours = find_contours(label_, 0.5)

#绘制轮廓
fig, (ax0,ax1) = plt.subplots(1,2,figsize=(16,16))
ax0.imshow(sure_fg,plt.cm.gray)
ax1.imshow(sure_fg,plt.cm.gray)

for n, contour in enumerate(contours):
    ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
ax1.axis('image')
ax1.set_xticks([])
ax1.set_yticks([])
plt.show()
print('总共有多少个⭕️：',len(contours))
# #
# #
# # contours = find_contours(distance, 0.5)
#
# #绘制轮廓
# fig, (ax0,ax1) = plt.subplots(1,2,figsize=(16,16))
# ax0.imshow(binary,plt.cm.gray)
# ax1.imshow(binary,plt.cm.gray)
#
# for n, contour in enumerate(contours):
#     ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
# ax1.axis('image')
# ax1.set_xticks([])
# ax1.set_yticks([])
# plt.show()
# print('总共有多少个⭕️：',len(contours))

# imshow(label_)
# plt.show()
# print(len(label_),label_.shape)
# #
# # label_ = label_[:,0]
# # plt.figure('hist_plot')
# #
# # arr = label_.flatten()
# # plt.hist(arr,256)
# label_ = cv2.cvtColor(label_,cv2.COLOR_BayerBG2GRAY)
# imshow(label_)


# contours = find_contours(label_, 0.5)
#
# #绘制轮廓
# fig, (ax0,ax1) = plt.subplots(1,2,figsize=(16,16))
# ax0.imshow(binary,plt.cm.gray)
# ax1.imshow(binary,plt.cm.gray)
#
# for n, contour in enumerate(contours):
#     ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
# ax1.axis('image')
# ax1.set_xticks([])
# ax1.set_yticks([])
# plt.show()
# print('总共有多少个⭕️：',len(contours))