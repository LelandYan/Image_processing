# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/15 21:46'


import numpy as np
import cv2
import random

# if __name__ == '__main__':
#
#     img = cv2.imread('the_first_image_cutting_processing.jpg',4)
#     h, w = img.shape[:2]
#     mask = np.zeros((h+2, w+2), np.uint8)
#     seed_pt = None
#     fixed_range = True
#     connectivity = 4
#
#     def update(dummy=None):
#         if seed_pt is None:
#             cv2.imshow('floodfill', img)
#             return
#         flooded = img.copy()
#         mask[:] = 1
#         lo = cv2.getTrackbarPos('lo', 'floodfill')
#         hi = cv2.getTrackbarPos('hi', 'floodfill')
#         flags = connectivity
#         if fixed_range:
#             flags |= cv2.FLOODFILL_FIXED_RANGE
#
#         cv2.floodFill(flooded, mask, seed_pt, (random.randint(0,255), random.randint(0,255), random.randint(0,255)), (lo,)*3, (hi,)*3, flags)
#
#         cv2.circle(flooded, seed_pt, 2, (0, 0, 255), -1)#选定基准点用红色圆点标出
#         cv2.namedWindow('floodfill', 0)
#         cv2.imshow('floodfill', flooded)
#
#
#     def onmouse(event, x, y, flags, param):#鼠标响应函数
#         global seed_pt
#         if flags & cv2.EVENT_FLAG_LBUTTON:#鼠标左键响应，选择漫水填充基准点
#             seed_pt = x, y
#             update()
#
#     update()
#     #cv2.setMouseCallback('floodfill', onmouse)
#     #cv2.createTrackbar('lo', 'floodfill', 20, 255, update)
#     #cv2.createTrackbar('hi', 'floodfill', 20, 255, update)
#
#     while True:
#         ch = 0xFF & cv2.waitKey()
#         if ch == 27:
#             break
#         if ch == ord('f'):
#             fixed_range = not fixed_range #选定时flags的高位比特位0，也就是邻域的选定为当前像素与相邻像素的的差，这样的效果就是联通区域会很大
#             # print 'using %s range' % ('floating', 'fixed')[fixed_range]
#             update()
#         if ch == ord('c'):
#             connectivity = 12-connectivity #选择4方向或则8方向种子扩散
#             # print 'connectivity =', connectivity
#             update()
#     cv2.destroyAllWindows()


im_in = cv2.imread("binary_first1111.jpg", cv2.IMREAD_GRAYSCALE)

# Threshold.
# Set values equal to or above 220 to 0.
# Set values below 220 to 255.

th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV)

# Copy the thresholded image.
im_floodfill = im_th.copy()

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = im_th.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0, 0), 255)
# Display images.
cv2.imwrite("edge_processing11111.jpg",im_floodfill)
# cv2.imshow("Thresholded Image", im_th)
# cv2.imshow("Floodfilled Image", im_floodfill)
# cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
cv2.namedWindow('im_floodfill', 0)
cv2.imshow("im_floodfill", im_floodfill)

cv2.waitKey(0)
cv2.destroyAllWindows()
#
# cv2.waitKey(0)

