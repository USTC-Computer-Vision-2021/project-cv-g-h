基于图像拼接的A Look Into the Past
===
成员及分工
---
 * 龚愉皓 PB18061299
   * 调试代码
   * 搜集素材
   * 撰写报告
 * 黄文强 PB18061271
   * 调试代码
   * 调研整理
   * 撰写报告

问题描述
--
* 初衷和动机<br>
  现代社会的发展日新月异，经济快速发展，城市面貌也飞速变化，如果能把以前的城市建筑，恢复到现在的城市中，就能有一种“昨日再现”的感觉，这就是我们选择“A Look Into the Past”的动机，新旧交替的视觉冲击会有不一样的体验。
* 创意描述<br>
  我们使用了莫斯科和美国国会大厦作为素材，这两对照片都是在同一角度对同一建筑物拍摄的，我们实现了把以前的照片嵌入现在的照片中，能看出面貌的巨大变化。
* 抽象模型<br>
  这个过程可以抽象成获取目标区域和图像拼接。

原理分析
--
* 特征点捕捉 (Interest Point Detection)<br>
  拍摄两张场景有重合的照片，建立了一个图像金字塔，在不同尺度寻找Harris关键点，并根据它捕捉特征点。
* 自适应非极大值抑制 (Adaptive Non-Maximal Suppression)<br>
  由于上一步得到的关键点很多，直接计算会导致很大的运算量，也会增加误差。接下去就要去除其中绝大部分的关键点，仅保留一些特征明显点，且让关键点在整幅图像内分布均匀。
* 关键点的描述 (Feature Descriptor)<br>
  对图像做适度的高斯模糊，以关键点为中心，取一部分像素的区域。将该区域降采样，一个向量。对向量做归一化处理。每个关键点都用一个向量表示，于是每幅图像分别得到了一个特征矩阵。<br>
* 关键点匹配<br>
  从两幅图片的特征点中筛选出配对的点，每次选择欧氏距离最小的一对点进行配对，关键点的匹配使用Random Sample Consensus (RANSAC) 算法。以一幅图像为基准，每次从中随机选择8个点，在另一幅图像中找出配对的8个点。用8对点计算得到一个homography，将基准图中剩余的特征点按照homography变换投影到另一幅图像，统计配对点的个数。重复上述步骤2000次，得到准确配对最多的一个homography。
* 新图像的合成<br>
  在做图像投影前，要先新建一个空白画布。比较投影后两幅图像的2维坐标的上下左右边界，选取各个方向边界的最大值作为新图像的尺寸。同时，计算得到两幅图像的交叉区域。在两幅图像的交叉区域，按照cross dissolve的方法制作，3个通道的像素值再次区间内递减（递升）。

代码实现
--
1.加载两个图像，转换为double及gray型。<br>
2.运用harris子函数检测两个图像中的特征点。<br>
3.在两个图像中的每个关键点周围提取固定大小的补丁，以及简单地通过将每个补丁中的像素值“展平”为一维向量来形成描述符。<br>
4.计算一幅图像中的每个描述符与另一幅图像中的每个描述符之间的距离。<br>
5.根据上面获得的成对描述符距离矩阵选择假定的匹配项。<br>
6.运行 RANSAC 以估计 (1) 仿射变换和 (2) 将一个图像映射到另一个图像的单应性。<br>
7.使用估计的变换将一个图像扭曲到另一个图像上。<br>
8.创建一个足够大的新图像以容纳全景图并将两个图像合成到其中。<br>

效果展示
--
* 国会大厦<br>
![](https://github.com/USTC-Computer-Vision-2021/project-cv-g-h/blob/main/Project/input/1.jpg)
![](https://github.com/USTC-Computer-Vision-2021/project-cv-g-h/blob/main/Project/input/2.jpg)
![](https://github.com/USTC-Computer-Vision-2021/project-cv-g-h/blob/main/Project/output/result.jpg)<br>
* 莫斯科<br>
下左图是20年前的莫斯科，下中图是现在的莫斯科，下右图是实现的效果。<br>
![效果图](https://github.com/USTC-Computer-Vision-2021/project-cv-g-h/blob/main/exp.png)

工程结构
--
```
.
├── code
│   ├── ada_nonmax_suppression.m
│   ├── blend.m
│   ├── dist2.m
│   ├── getFeatureDescriptor.m
│   ├── getHomographyMatrix.m
│   ├── getNewSize.m
│   ├── harris.m
│   ├── image_stitching.m
│   ├── main.m
│   └── ransacfithomography.m
├── input
│   ├── 1.jpg
│   └── 2.jpg
└── output
    └── result.jpg
```

运行说明
--
运行main.m，由于子函数中存在随机性，若出现报错情况，重新运行即可。
