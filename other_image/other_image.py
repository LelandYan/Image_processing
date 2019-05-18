# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/17 18:55'

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import skimage as sm
from skimage import morphology
from skimage.feature import peak_local_max

image = cv2.imread("./raw_data/1.jpg")
kernel_sharpen_1 = np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]])
kernel_sharpen_2 = np.array([
        [1,1,1],
        [1,-7,1],
        [1,1,1]])
kernel_sharpen_3 = np.array([
        [-1,-1,-1,-1,-1],
        [-1,2,2,2,-1],
        [-1,2,8,2,-1],
        [-1,2,2,2,-1],
        [-1,-1,-1,-1,-1]])/8.0

output_1 = cv2.filter2D(image,-1,kernel_sharpen_1)
output_2 = cv2.filter2D(image,-1,kernel_sharpen_2)
output_3 = cv2.filter2D(image,-1,kernel_sharpen_3)
#显示锐化效果
# cv2.namedWindow('Original Image',  cv2.WINDOW_NORMAL)
# cv2.imwrite('Original_Image1.jpg',image)
# # cv2.namedWindow('sharpen_1 Image', cv2.WINDOW_NORMAL)
# cv2.imwrite('./out_data/sharpen_1_Image1.jpg',output_1)
# # cv2.namedWindow('sharpen_2 Image', cv2.WINDOW_NORMAL)
# cv2.imwrite('./out_data/sharpen_2_Image1.jpg',output_2)
# # cv2.namedWindow('sharpen_3 Image', cv2.WINDOW_NORMAL)
# cv2.imwrite('./out_data/sharpen_3_Image1.jpg',output_3)


output_1 = cv2.cvtColor(output_1, cv2.COLOR_RGB2GRAY)  #把输入图像灰度化
output_2 = cv2.cvtColor(output_2, cv2.COLOR_RGB2GRAY)  #把输入图像灰度化
output_3 = cv2.cvtColor(output_3, cv2.COLOR_RGB2GRAY)  #把输入图像灰度化
# cv2.namedWindow('sharpen_1 Image', cv2.WINDOW_NORMAL)
# cv2.imwrite('./out_data/gray_sharpen_1_Image1.jpg',output_1)
# # cv2.namedWindow('sharpen_2 Image', cv2.WINDOW_NORMAL)
# cv2.imwrite('./out_data/gray_sharpen_2_Image1.jpg',output_2)
# # cv2.namedWindow('sharpen_3 Image', cv2.WINDOW_NORMAL)
# cv2.imwrite('./out_data/gray_sharpen_3_Image1.jpg',output_3)


plt.hist(output_1.ravel(),256)
plt.show()
ret, binary = cv2.threshold(output_1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)#TRIANGLE法,，全局自适应阈值, 参数0可改为任意数字但不起作用，适用于单个波峰
print("阈值：%s" % ret)
# cv2.namedWindow("binary0", cv2.WINDOW_NORMAL)
# #cv.imwrite("binary_first11.jpg", binary)
# cv2.imshow("binary0", binary)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
rows,cols = output_1.shape
labels = np.zeros([rows,cols])
for i in range(rows):
    for j in range(cols):
        if(output_1[i,j] > 59):
                labels[i,j] = 1
        else:
            labels[i,j] = 0

# cv2.namedWindow("binary0", cv2.WINDOW_NORMAL)
# #cv.imwrite("binary_first11.jpg", binary)
# cv2.imshow("binary0", labels)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
distance = ndi.distance_transform_edt(labels) #距离变换
# min_distance：最小的像素在2×min_distance + 1区分离（即峰峰数至少min_distance分隔）。找到峰值的最大数量，使用min_distance = 1。
# exclude_border：不排除峰值在图像的边界
# indices：False会返回和数组相同大小的布尔数组，为True时，会返回峰值的坐标
local_maxi = peak_local_max(distance, exclude_border = 0,min_distance = 12,indices=False,
                                   footprint=np.ones((10, 10)),labels=labels) #寻找峰值
markers = ndi.label(local_maxi)[0] #初始标记点
label_ =morphology.watershed(-distance, markers, mask=labels) #基于距离变换的分水岭算法

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
axes = axes.ravel()
ax0, ax1, ax2, ax3 = axes

# ax0.imshow(labels, cmap=plt.cm.gray)#, interpolation='nearest')
# ax0.set_title("Original")
# ax1.imshow(-distance, cmap=plt.cm.jet, interpolation='nearest')
# ax1.set_title("Distance")
# # ax2.imshow(sm.dilation(markers,sm.square(10)), cmap=plt.cm.spectral, interpolation='nearest')
# # ax2.set_title("Markers")
# ax3.imshow(label_, cmap=plt.cm.spectral, interpolation='nearest')
# ax3.set_title("Segmented")
#
# for ax in axes:
#     ax.axis('off')

# fig.tight_layout()
# plt.show()
import math
err = []
import math
err = []
for i in range(labels.shape[0]):
    h1,w1 =  labels[i][0],labels[i][1]
    if i in err:
        continue
    for j in range(i+1,labels.shape[0]):
        h2,w2 =  labels[j][0],labels[j][1]
        ab = math.sqrt(math.pow(abs(h2-h1), 2) + math.pow(abs(w2-w1), 2))
        if ab <= 10:
#             print 'error:' , x_y[i],' and ', x_y[j],'i,j = ',i,j
            err.append(j)
new_x_y = []
for i in range(len(labels)):
    if i not in err:
        new_x_y.append(labels[i])
print('一共有',len(new_x_y),'个圈')


# def threshold_demo(image):
#     gray = image
#     # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #把输入图像灰度化
#     #直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
#     ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
#     print("threshold value %s"%ret)
#     cv2.namedWindow("binary0", cv2.WINDOW_NORMAL)
#     #cv.imwrite("binary_first11.jpg", binary)
#     cv2.imshow("binary0", binary)
#
# #局部阈值
# def local_threshold(image):
#     gray = image
#     # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #把输入图像灰度化
#     #自适应阈值化能够根据图像不同区域亮度分布，改变阈值
#     binary =  cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 10)
#     cv2.namedWindow("binary1", cv2.WINDOW_NORMAL)
#     #cv.imwrite("binary_first22.jpg", binary)
#     cv2.imshow("binary1", binary)
#
# #用户自己计算阈值
# def custom_threshold(image):
#     gray = image
#     # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #把输入图像灰度化
#     h, w =gray.shape[:2]
#     m = np.reshape(gray, [1,w*h])
#     mean = m.sum()/(w*h)
#     print("mean:",mean)
#     ret, binary =  cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
#     #cv.imwrite("binary_first33.jpg", binary)
#     cv2.namedWindow("binary2", cv2.WINDOW_NORMAL)
#     cv2.imshow("binary2", binary)
#
# # src = cv2.imread(output_1)
# src = output_3
# cv2.namedWindow('input_image', cv2.WINDOW_NORMAL) #设置为WINDOW_NORMAL可以任意缩放
# cv2.imshow('input_image', src)
#
# threshold_demo(src)
# local_threshold(src)
# custom_threshold(src)
# cv2.waitKey(0)
# cv2.destroyAllWindows()