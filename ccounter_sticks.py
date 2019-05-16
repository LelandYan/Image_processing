# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/15 19:01'
import cv2
import numpy as np
# 进行图像的填充
# img = cv2.imread("the_first_image_cutting_processing.jpg")


def FillHole(imgPath, SavePath):
    im_in = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("im_in.png", im_in)
    # 复制 im_in 图像
    im_floodfill = im_in.copy()

    # Mask 用于 floodFill，官方要求长宽+2
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # floodFill函数中的seedPoint对应像素必须是背景
    isbreak = False
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if (im_floodfill[i][j] == 0):
                seedPoint = (i, j)
                isbreak = True
                break
        if (isbreak):
            break

    # 得到im_floodfill 255填充非孔洞值
    cv2.floodFill(im_floodfill, mask, seedPoint, 255)

    # 得到im_floodfill的逆im_floodfill_inv
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    im_out = im_in | im_floodfill_inv

    # 保存结果
    cv2.imwrite(SavePath, im_out)

# FillHole("the_first_image_cutting_processing.jpg",'test_image.jpg')




# 指定颜色替换
# def fill_image(image):
#     copyImage = image.copy()  # 复制原图像
#     h, w = image.shape[:2]  # 读取图像的宽和高
#     mask = np.zeros([h + 2, w + 2], np.uint8)  # 新建图像矩阵  +2是官方函数要求
#     cv2.floodFill(copyImage, mask, (0, 80), (0, 100, 255), (100, 100, 50), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
#     cv2.imshow("填充", copyImage)


# img = cv2.imread("the_first_image_cutting_processing.jpg")
# fill_image(img)
# img_rgb = cv2.imread('processing_image2.jpg')
img_rgb = cv2.imread('the_first_image_cutting_processing.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('icon.jpg', 0)
h, w = template.shape[:2]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
# 取匹配程度大于%80的坐标
loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):  # *号表示可选参数
    bottom_right = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img_rgb, pt, bottom_right, (0, 0, 255), 2)

cv2.namedWindow('im_out', 0)
cv2.imshow("im_out", img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
