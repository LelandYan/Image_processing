## 第一张图片处理

原图为：

![](C:\Users\lenovo\Desktop\Image_processing\processing_steps.assets\1.JPG)

下面为处理过程：

1. 灰度化处理

2. 开运算操作 其中包含两种操作

   1. 腐蚀操作 --迭代5次
   2. 扩展操作 --迭代5次

3. 灰度进一步处理分别使用开运算，闭运算，梯度运算，顶帽（Top-hat）运算(MORPH_TOPHAT)是计算开运算结果图与原始图像之差，黑帽（Black Hot）运算(MORPH_BLACKHAT)是计算闭运算结果图与原始图像之差，按行从左到右。

   ![](C:\Users\lenovo\Desktop\Image_processing\processing_steps.assets\first_image.png)

4. 由上步可得梯度处理的效果较好，所以使用梯度处理得到效果最好的灰度图。这里发现沙子对图像的影响较大，所以要去图像进行轮廓检测。这里轮廓检测是沙子。

   ![](C:\Users\lenovo\Desktop\Image_processing\processing_steps.assets\gradient_image.jpg)

5. 经过轮廓检测后的灰度图，这里用红色矩形画出

   ![](C:\Users\lenovo\Desktop\Image_processing\processing_steps.assets\processing_edge_cutting-1557993288749.png)

6. 在原图进行切割的效果图

   ![](C:\Users\lenovo\Desktop\Image_processing\processing_steps.assets\edg_cutting.png)

7. 原图切割后得到

   ![](C:\Users\lenovo\Desktop\Image_processing\processing_steps.assets\processing_image2.jpg)

8. 对切割后的图片进行灰度化处理处理过程见 1-3，然后可以效果图

   ![](C:\Users\lenovo\Desktop\Image_processing\processing_steps.assets\the_first_image_cutting_processing.jpg)

9. 对图像进行二值化处理 对上步得到的灰度图进行滤波处理得到二值化图
   
   1. 使用全局阀值
   
      ![](C:\Users\lenovo\Desktop\Image_processing\processing_steps.assets\binary_first1.jpg)
   
   2. 使用局部阀值
   
      ![](C:\Users\lenovo\Desktop\Image_processing\processing_steps.assets\binary_first2.jpg)
   
   3. 使用根据亮度自定义阈值
   
      ![](C:\Users\lenovo\Desktop\Image_processing\processing_steps.assets\binary_first3.jpg)
   
   4. 使用滤波法，将rgb限制为220-255之间，对图像进行再次处理
   
      ![](C:\Users\lenovo\Desktop\Image_processing\processing_steps.assets\edge_processing1.jpg)