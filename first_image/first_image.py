# _*_ coding: utf-8 _*_
__author__ = 'LelandYan'
__date__ = '2019/5/16 19:32'
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
from pylab import mpl
import skimage
# 防止中文乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']


class processing_image:
    def __init__(self, filename="./raw_data/1.jpg", output="./out_data"):
        self.filename = filename
        self.output = output

    def op_gray_to_four_type(self, kernel=(9, 9), erode_iter=5, dilate_iter=5):

        img = cv2.imread(self.filename)
        # gray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # erode dilate
        closed = cv2.erode(img, None, iterations=erode_iter)
        img = cv2.dilate(closed, None, iterations=dilate_iter)

        kernel = np.ones(kernel, np.uint8)
        # open operation
        img_open = cv2.morphologyEx(img, op=cv2.MORPH_OPEN, kernel=kernel)
        # close operation
        img_close = cv2.morphologyEx(img, op=cv2.MORPH_CLOSE, kernel=kernel)
        # gradient operation
        img_grad = cv2.morphologyEx(img, op=cv2.MORPH_GRADIENT, kernel=kernel)
        # tophat operation
        img_tophat = cv2.morphologyEx(img, op=cv2.MORPH_TOPHAT, kernel=kernel)
        # blackhat operation
        img_blackhat = cv2.morphologyEx(img, op=cv2.MORPH_BLACKHAT, kernel=kernel)
        # Plot the images
        images = [img, img_open, img_close, img_grad,
                  img_tophat, img_blackhat]
        names = ["raw_img", "img_open", "img_close", "img_grad", "img_tophat", "img_blackhat"]
        cv2.imwrite(self.output+"/gradient_image1.jpg",img_grad)
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 15))
        for ind, p in enumerate(images):
            ax = axs[ind // 3, ind % 3]
            ax.imshow(p, cmap='gray')
            ax.set_title(names[ind])
            ax.axis('off')
        plt.show()

    def op_first_to_three_type(self, flag=False):
        # 全局阈值
        def threshold_demo(image):
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 把输入图像灰度化
            # 直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
            ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
            if flag:
                cv2.imwrite(self.output + "/global_binary_first1.jpg", binary)
            return binary

        # 局部阈值
        def local_threshold(image):
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 把输入图像灰度化
            # 自适应阈值化能够根据图像不同区域亮度分布，改变阈值
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)

            if flag:
                cv2.imwrite(self.output + "/local_binary_first1.jpg", binary)
            return binary

        # 用户自己计算阈值
        def custom_threshold(image):
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 把输入图像灰度化
            h, w = gray.shape[:2]
            m = np.reshape(gray, [1, w * h])
            mean = m.sum() / (w * h)
            ret, binary = cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
            if flag:
                cv2.imwrite(self.output + "/custom_binary_first1.jpg", binary)
            return binary

        if flag:
            src = cv2.imread("./out_data/gray_cutting_image1.jpg")
        else:
            src = cv2.imread(self.filename)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        global_scr = threshold_demo(src)
        local_scr = local_threshold(src)
        custom_src = custom_threshold(src)
        images = [src, global_scr, local_scr,
                  custom_src]
        names = ["src", "global_scr", "local_scr", "custom_src"]
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        for ind, p in enumerate(images):
            ax = axs[ind // 2, ind % 2]
            ax.imshow(p, cmap='gray')
            ax.set_title(names[ind])
            ax.axis('off')
        plt.show()

    def op_cutting_image(self):
        raw_img = cv2.imread(self.filename)
        img = cv2.imread("./out_data/gradient_image1.jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 7, sigmaSpace=75, sigmaColor=75)
        ret, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        closed = cv2.dilate(binary, None, iterations=130)
        closed = cv2.erode(closed, None, iterations=127)

        _, contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(contours, key=cv2.contourArea, reverse=True)[0]

        # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        # draw a bounding box arounded the detected barcode and display the image
        draw_img = cv2.drawContours(raw_img.copy(), [box], -1, (0, 0, 255), 3)

        h, w, _ = img.shape
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        hight = y2 - y1
        width = x2 - x1
        crop_img = img[0:h - hight, x1:x1 + width]
        raw_img = raw_img[0:h - hight, x1:x1 + width]
        cv2.imwrite(self.output + "/raw_draw_image1.jpg", draw_img)
        cv2.imwrite(self.output + "/raw_cutting_image1.jpg", raw_img)
        cv2.imwrite(self.output + "/gray_cutting_image1.jpg", crop_img)

    def op_edge_test(self):
        def gray_dege_test():
            img = cv2.imread("./out_data/gradient_image1.jpg")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 0)
            ret, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
            closed = cv2.dilate(binary, None, iterations=110)
            closed = cv2.erode(closed, None, iterations=120)

            _, contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
            plt.imshow(img)
            plt.show()
            cv2.imwrite(self.output + "/gray_edge_test.jpg", img)

        def fourier_edge_test():
            img = cv2.imread('./out_data/gradient_image1.jpg', 0)
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)

            rows, cols = img.shape
            crow, ccol = int(rows / 2), int(cols / 2)
            for i in range(crow - 30, crow + 30):
                for j in range(ccol - 30, ccol + 30):
                    fshift[i][j] = 0.0
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)  # 进行高通滤波
            # 取绝对值
            img_back = np.abs(img_back)
            plt.subplot(121), plt.imshow(img, cmap='gray')  # 因图像格式问题，暂已灰度输出
            plt.title('Input Image'), plt.xticks([]), plt.yticks([])
            # 先对灰度图像进行伽马变换，以提升暗部细节
            rows, cols = img_back.shape
            gamma = copy.deepcopy(img_back)
            rows = img.shape[0]
            cols = img.shape[1]
            for i in range(rows):
                for j in range(cols):
                    gamma[i][j] = 5.0 * pow(gamma[i][j], 0.34)  # 0.34这个参数是我手动调出来的，根据不同的图片，可以选择不同的数值
            # 对灰度图像进行反转

            for i in range(rows):
                for j in range(cols):
                    gamma[i][j] = 255 - gamma[i][j]

            plt.subplot(122), plt.imshow(gamma, cmap='gray')
            plt.title('Result in HPF'), plt.xticks([]), plt.yticks([])
            cv2.imwrite(self.output + "/fourier_edge_test_image1.jpg", gamma)
            plt.show()

        def canny_edge_test():
            img = cv2.imread('./out_data/gradient_image1.jpg', 0)
            edges = cv2.Canny(img, 100, 200)

            plt.subplot(121), plt.imshow(img, cmap='gray')
            plt.title('original'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(edges, cmap='gray')
            plt.title('edge'), plt.xticks([]), plt.yticks([])
            cv2.imwrite(self.output + "/canny_edge_test_image1.jpg", edges)
            plt.show()

        gray_dege_test()
        fourier_edge_test()
        canny_edge_test()

    def op_trans_plot(self):
        im_in = cv2.imread("./out_data/custom_binary_first1.jpg", cv2.IMREAD_GRAYSCALE)
        th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV)

        # Copy the thresholded image.
        im_floodfill = im_th.copy()

        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = im_th.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        cv2.imwrite(self.output + "/edge_processing1.jpg", im_floodfill)

    def op_counter(self):
        ob1 = cv2.imread("./out_data/edge_processing1.jpg", cv2.IMREAD_GRAYSCALE)
        # ob1 = cv2.dilate(ob1, None, iterations=2)
        ob1 = cv2.bilateralFilter(ob1, 7, sigmaSpace=70, sigmaColor=70)
        ob1 = cv2.erode(ob1, None, iterations=2) # 1 # 2
        ob1 = cv2.dilate(ob1, None, iterations=2)
        ob2 = cv2.imread("./out_data/icon4.jpg", cv2.IMREAD_GRAYSCALE)
        # ob2 = cv2.bilateralFilter(ob2, 7, sigmaSpace=60, sigmaColor=60)
        ob2 = cv2.erode(ob2, None, iterations=1)
        # ob2 = cv2.dilate(ob2, None, iterations=1)
        # orb = cv2.xfeatures2d.SURF_create()
        orb = cv2.xfeatures2d.SIFT_create()
        keyp1, desp1 = orb.detectAndCompute(ob1, None)
        keyp2, desp2 = orb.detectAndCompute(ob2, None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desp1, desp2, k=2)
        matchesMask = [[0, 0] for i in range(len(matches))]
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]
        # 如果第一个邻近距离比第二个邻近距离的0.7倍小，则保留
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matchesMask, flags=0)
        img3 = cv2.drawMatchesKnn(ob1, keyp1, ob2, keyp2, matches, None, **draw_params)
        a = len(keyp1) // len(keyp2)
        plt.figure(figsize=(8, 8))
        plt.subplot(211)
        plt.imshow(img3)
        plt.subplot(212)
        plt.text(0.5, 0.6, "the number of sticks:" + str(a), size=30, ha="center", va="center")
        plt.axis('off')
        plt.show()
        cv2.imwrite(self.output+"/counter_sticks_image1.jpg", img3)

if __name__ == '__main__':
    ob = processing_image()
    ob.op_gray_to_four_type()
    ob.op_first_to_three_type()
    # ob.op_cutting_image()
    ob.op_edge_test()
    ob.op_trans_plot()
    ob.op_first_to_three_type(flag=True)
    ob.op_counter()
