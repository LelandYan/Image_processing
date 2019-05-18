# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/15 19:01'
# import cv2
# import numpy as np
# 进行图像的填充
# img = cv2.imread("the_first_image_cutting_processing.jpg")


# def FillHole(imgPath, SavePath):
#     im_in = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
#     cv2.imwrite("im_in.png", im_in)
#     # 复制 im_in 图像
#     im_floodfill = im_in.copy()
#
#     # Mask 用于 floodFill，官方要求长宽+2
#     h, w = im_in.shape[:2]
#     mask = np.zeros((h + 2, w + 2), np.uint8)
#
#     # floodFill函数中的seedPoint对应像素必须是背景
#     isbreak = False
#     for i in range(im_floodfill.shape[0]):
#         for j in range(im_floodfill.shape[1]):
#             if (im_floodfill[i][j] == 0):
#                 seedPoint = (i, j)
#                 isbreak = True
#                 break
#         if (isbreak):
#             break
#
#     # 得到im_floodfill 255填充非孔洞值
#     cv2.floodFill(im_floodfill, mask, seedPoint, 255)
#
#     # 得到im_floodfill的逆im_floodfill_inv
#     im_floodfill_inv = cv2.bitwise_not(im_floodfill)
#
#     # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
#     im_out = im_in | im_floodfill_inv
#
#     # 保存结果
#     cv2.imwrite(SavePath, im_out)
#
# # FillHole("the_first_image_cutting_processing.jpg",'test_image.jpg')
#
#
#
#
# # 指定颜色替换
# # def fill_image(image):
# #     copyImage = image.copy()  # 复制原图像
# #     h, w = image.shape[:2]  # 读取图像的宽和高
# #     mask = np.zeros([h + 2, w + 2], np.uint8)  # 新建图像矩阵  +2是官方函数要求
# #     cv2.floodFill(copyImage, mask, (0, 80), (0, 100, 255), (100, 100, 50), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
# #     cv2.imshow("填充", copyImage)
#
#
# # img = cv2.imread("the_first_image_cutting_processing.jpg")
# # fill_image(img)
# # img_rgb = cv2.imread('processing_image2.jpg')
# img_rgb = cv2.imread('the_first_image_cutting_processing.jpg')
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# template = cv2.imread('icon.jpg', 0)
# h, w = template.shape[:2]
#
# res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
# threshold = 0.8
# # 取匹配程度大于%80的坐标
# loc = np.where(res >= threshold)
#
# for pt in zip(*loc[::-1]):  # *号表示可选参数
#     bottom_right = (pt[0] + w, pt[1] + h)
#     cv2.rectangle(img_rgb, pt, bottom_right, (0, 0, 255), 2)
#
# cv2.namedWindow('im_out', 0)
# cv2.imshow("im_out", img_rgb)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import numpy as np
import matplotlib.pyplot as plt






person_1 = cv2.imread("edge_processing11111.jpg", cv2.IMREAD_GRAYSCALE)
person_1 = cv2.bilateralFilter(person_1, 7, sigmaSpace=70, sigmaColor=70)
person_1 = cv2.erode(person_1, None, iterations=2)  # 1 # 2
person_1 = cv2.dilate(person_1, None, iterations=2)


person_2 = cv2.imread("icon12.png", cv2.IMREAD_GRAYSCALE)
person_2 = cv2.erode(person_2, None, iterations=1)
# person_2 = cv2.erode(person_2, None, iterations=5)
# person_2 = cv2.dilate(person_2, None, iterations=2)
# orb = cv2.xfeatures2d.SIFT_create()
# orb = cv2.xfeatures2d.SURF_create()
orb = cv2.xfeatures2d.SIFT_create()
keyp1 ,desp1= orb.detectAndCompute(person_1, None)
keyp2 ,desp2= orb.detectAndCompute(person_2, None)
print(len(keyp2))
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(desp1,desp2,k=2)
print(len(matches))
matchesMask = [[0,0] for i in range(len(matches))]
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
         matchesMask[i] = [1, 0]
#如果第一个邻近距离比第二个邻近距离的0.7倍小，则保留
print(len(keyp1)//len(keyp2))
a = len(keyp1)//len(keyp2)
draw_params = dict(matchColor = (0,255,0),singlePointColor = (255,0,0),matchesMask = matchesMask,flags = 0)


img3 = cv2.drawMatchesKnn(person_1,keyp1,person_2,keyp2,matches,None,**draw_params)
plt.figure(figsize=(8,8))
plt.subplot(211)
plt.imshow(img3)
plt.subplot(212)
plt.text(0.5,0.6,"the number of sticks:"+str(a),size=30,ha="center",va="center")
plt.axis('off')
plt.show()










# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(desp1, desp2)
# matches = sorted(matches, key = lambda x:x.distance)
# matchpic = cv2.drawMatches(person_1, keyp1, person_2, keyp2, matches[:25],
#                            person_2,flags=2)
# plt.imshow(matchpic)
# plt.show()




# keypoint_1 = cv2.drawKeypoints(image=person_1, outImage=person_1,keypoints=keyp1,
#                               flags=4, color=(255,0,0))
# keypoint_2 = cv2.drawKeypoints(image=person_2, outImage=person_2,keypoints=keyp2,
#                               flags=4, color=(255,0,0))
# plt.figure(figsize=(11,7))
# plt.imshow(keypoint_1)
# plt.show()
# plt.figure(figsize=(11,7))
# plt.imshow(person_1)
# # plt.imshow(keypoint_2)
# plt.show()
# plt.figure(figsize=(11,7))
# plt.imshow(person_1)
# # plt.imshow(person_2)
# plt.show()
# 对目标进行模糊化处理
# closed = cv2.dilate(closed, None, iterations=3)
# closed = cv2.erode(closed, None, iterations=10)
# cv2.namedWindow('img', 0)
# cv2.imshow('img', closed)
# cv2.imwrite("edge_processing2.jpg",closed)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des1,des2)
# matches = sorted(matches, key = lambda x:x.distance)
# matchImg = cv2.drawMatches(person_1, kp1, person_2, kp2, matches[:30], person_2,
#                           flags=2)
# plt.figure(figsize=(15,8))
# plt.imshow(matchImg)
# plt.show()