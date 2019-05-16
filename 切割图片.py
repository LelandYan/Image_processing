# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/15 17:49'
import cv2
import matplotlib.pyplot as plt
import numpy as np
raw_img = cv2.imread('1.jpg')
img = cv2.imread("gradient_image.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.bilateralFilter(gray, 7, sigmaSpace = 75, sigmaColor =75)
ret, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
closed = cv2.dilate(binary, None, iterations=200)
closed = cv2.erode(closed, None, iterations=200)

_, contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
c = sorted(contours, key=cv2.contourArea, reverse=True)[0]

# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))
#
# draw a bounding box arounded the detected barcode and display the image
draw_img = cv2.drawContours(raw_img.copy(), [box], -1, (0, 0, 255), 3)
# cv2.namedWindow('draw_img', 0)
# cv2.imshow("draw_img", draw_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.drawContours(img, contours, -1, (0, 0, 255), 3)



h,w,_ = img.shape
Xs = [i[0] for i in box]
Ys = [i[1] for i in box]
x1 = min(Xs)
x2 = max(Xs)
y1 = min(Ys)
y2 = max(Ys)
hight = y2 - y1
width = x2 - x1
crop_img= img[0:h-hight, x1:x1+width]

cv2.namedWindow('crop_img', 0)
cv2.imshow('crop_img', draw_img)
# cv2.imwrite("processing_image2.jpg",crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

