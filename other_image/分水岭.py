# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/19 10:25'

import cv2 as cv
import numpy as np

def water_shed(image):
    #1 去噪，灰度，二值化
    blurred = cv.pyrMeanShiftFiltering(image,10,30)
    blurred=cv.bilateralFilter(image,0,50,5)
    # cv.imshow('blurred',blurred)
    gray=cv.cvtColor(blurred,cv.COLOR_BGR2GRAY)
    ret,binary=cv.threshold(gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
    # cv.imshow('binary',binary)
    #2. mophology 开操作去除噪点
    kernel=cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    open_binary=cv.morphologyEx(binary,cv.MORPH_OPEN,kernel,iterations=2) #mophology binary,2次开操作
    # cv.imshow('1-open-op',open_binary)
    dilate_bg=cv.dilate(open_binary,kernel,iterations=3) #3次膨胀
    # cv.imshow('2-dilate-op',dilate_bg)
    #3.distance transform
    # DIST_L1:曼哈顿距离，DIST_L2：欧氏距离,masksize:跟卷积一样
    dist=cv.distanceTransform(open_binary,cv.DIST_L2,3) #？？
    dist_norm=cv.normalize(dist,0,1.0,cv.NORM_MINMAX)# 0-1之间标准化
    # cv.imshow('3-distance-t',dist_norm*50)

    ret,surface=cv.threshold(dist,dist.max()*0.65,255,cv.THRESH_BINARY)
    # cv.imshow('4-surface',surface)

    #4计算marker
    surface_fg=np.uint8(surface) #计算前景
    unknown=cv.subtract(dilate_bg,surface_fg) #计算未知区域
    # cv.imshow('5-unknown',unknown)
    ret,markers=cv.connectedComponents(surface_fg) #通过计算cc，计算markers
    print(ret)
    # cv.imshow('6-markers',markers)

    #5 watershed 分水岭变换
    markers=markers+1 #用label进行控制
    markers[unknown==255]=0
    markers=cv.watershed(image,markers) #分水岭的地方就编程-1
    image[markers==-1]=[0,0,255]
    cv.imshow('7-result',image)

src = cv.imread("./raw_data/4.jpg")
# cv.imshow("gray_img", src)
cv.namedWindow("result", cv.WINDOW_NORMAL)
water_shed(src)
cv.waitKey(0)
cv.destroyAllWindows()