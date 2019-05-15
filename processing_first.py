# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/15 13:57'
import cv2
import matplotlib.pyplot as plt


img = cv2.imread("1.jpg")
# plt.imshow(img)
# plt.show()

img_rbg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# plt.imshow(img_rbg)
# # plt.show()
#
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# plt.imshow(img_gray, cmap = 'gray')
# # plt.show()
#
# fig, axs = plt.subplots(nrows = 1, ncols = 3,figsize = (20, 20))
#
# # 三色调叠加，成为彩色的图片
# for i in range(0,3):
#     ax = axs[i]
#     ax.imshow(img_rbg[:,:,i],cmap='gray')
#
# # plt.show()
# img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
# fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols =2, figsize = (20, 20))
# ax1.imshow(img_hsv)
# ax2.imshow(img_hls)

# plt.show()

# kernels = [5,11,17]
# fig,axs = plt.subplots(nrows=1,ncols=3,figsize=(20,20))
# for ind,s in enumerate(kernels):
#     img_blurred = cv2.blur(img_rbg,ksize=(s,s))
#     ax = axs[ind]
#     ax.imshow(img_blurred)
#     ax.axis('off')
# plt.show()
# img_0 = cv2.blur(img_rbg, ksize = (7, 7))
# img_1 = cv2.GaussianBlur(img_rbg, ksize = (7, 7), sigmaX = 0)
# img_2 = cv2.medianBlur(img_rbg, 7)
# img_3 = cv2.bilateralFilter(img_rbg, 7, sigmaSpace = 75, sigmaColor =75)
# # Plot the images
# images = [img_0, img_1, img_2, img_3]
# fig, axs = plt.subplots(nrows = 1, ncols = 4, figsize = (20, 20))
# for ind, p in enumerate(images):
#     ax = axs[ind]
#     ax.imshow(p)
#     ax.axis('off')
# plt.show()

# _, thresh_0 = cv2.threshold(img_rbg, 127, 255, cv2.THRESH_BINARY)
# _, thresh_1 = cv2.threshold(img_rbg, 127, 255, cv2.THRESH_BINARY_INV)
# _, thresh_2 = cv2.threshold(img_rbg, 127, 255, cv2.THRESH_TOZERO)
# _, thresh_3 = cv2.threshold(img_rbg, 127, 255, cv2.THRESH_TOZERO_INV)
# _, thresh_4 = cv2.threshold(img_rbg, 127, 255, cv2.THRESH_TRUNC)
# # Plot the images
# images = [img, thresh_0, thresh_1, thresh_2, thresh_3, thresh_4]
# fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (13, 13))
# for ind, p in enumerate(images):
#     ax = axs[ind//3, ind%3]
#     ax.imshow(p)
# plt.show()


# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # Adaptive Thresholding
# _, thresh_binary = cv2.threshold(img, thresh = 127, maxval = 255, type = cv2.THRESH_BINARY)
# adap_mean_2 = cv2.adaptiveThreshold(img, 255,
#                                     cv2.ADAPTIVE_THRESH_MEAN_C,
#                                     cv2.THRESH_BINARY, 7, 2)
# adap_mean_2_inv = cv2.adaptiveThreshold(img, 255,
#                                         cv2.ADAPTIVE_THRESH_MEAN_C,
#                                         cv2.THRESH_BINARY_INV, 7, 2)
# adap_mean_8 = cv2.adaptiveThreshold(img, 255,
#                                     cv2.ADAPTIVE_THRESH_MEAN_C,
#                                     cv2.THRESH_BINARY, 7, 8)
# adap_gaussian_8 = cv2.adaptiveThreshold(img, 255,
#                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                     cv2.THRESH_BINARY, 7, 8)
# images = [img, thresh_binary, adap_mean_2, adap_mean_2_inv,
#           adap_mean_8, adap_gaussian_8]
# fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (15, 15))
# for ind, p in enumerate(images):
#     ax = axs[ind%2, ind//2]
#     ax.imshow(p, cmap = 'gray')
#     ax.axis('off')
# plt.show()

# sobel_x = cv2.Sobel(img, cv2.CV_64F, dx = 1, dy = 0, ksize = 5)
# sobel_y = cv2.Sobel(img, cv2.CV_64F, dx = 0, dy = 1, ksize = 5)
# blended = cv2.addWeighted(src1=sobel_x, alpha=0.5, src2=sobel_y,
#                           beta=0.5, gamma=0)
# laplacian = cv2.Laplacian(img, cv2.CV_64F)
# images = [sobel_x, sobel_y, blended, laplacian]
# plt.figure(figsize = (20, 20))
# for i in range(4):
#     plt.subplot(1, 4, i+1)
#     plt.imshow(images[i], cmap = 'gray')
#     plt.axis('off')
# plt.show()

# import numpy as np
# kernel_0 = np.ones((9, 9), np.uint8)
# kernel_1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
# kernel_2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
# kernels = [kernel_0, kernel_1, kernel_2]
# # Plot the images
# plt.figure(figsize = (20, 20))
# for i in range(3):
#     img_copy = img.copy()
#     img_copy = cv2.erode(img_copy, kernels[i], iterations = 3)
#     plt.subplot(1, 3, i+1)
#     plt.imshow(img_copy)
#     plt.axis('off')
# plt.show()
#
# kernel = np.ones((9, 9), np.uint8)
# img_dilate = cv2.dilate(img, kernel, iterations = 3)
# plt.figure(figsize = (20, 10))
# plt.subplot(1, 2, 1); plt.imshow(img, cmap="gray")
# plt.subplot(1, 2, 2); plt.imshow(img_dilate, cmap="gray")
# plt.show()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Canny detection without blurring
# edges = cv2.Canny(image=img, threshold1=127, threshold2=127)
# plt.figure(figsize = (20, 20))
# plt.subplot(1, 2, 1); plt.imshow(img)
# plt.axis('off')
# plt.subplot(1, 2, 2); plt.imshow(edges)
# plt.axis('off')
# plt.show()
import numpy as np
med_val = np.median(img)
lower = int(max(0, .7*med_val))
upper = int(min(255, 1.3*med_val))

img_k5 = cv2.blur(img, ksize = (5, 5))
# Canny detection with different thresholds
edges_k5 = cv2.Canny(img_k5, threshold1 = lower, threshold2 = upper)
edges_k5_2 = cv2.Canny(img_k5, lower, upper+100)
# Blurring with ksize = 9
img_k9 = cv2.blur(img, ksize = (9, 9))
# Canny detection with different thresholds
edges_k9 = cv2.Canny(img_k9, lower, upper)
edges_k9_2 = cv2.Canny(img_k9, lower, upper+100)
# Plot the images
images = [edges_k5, edges_k5_2, edges_k9, edges_k9_2]
plt.figure(figsize = (20, 15))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i])
    plt.axis('off')
plt.show()