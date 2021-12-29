基于图像拼接的A Look Into the Past
===
成员及分工
---
 * 龚愉皓 PB18061299
   * 调试代码
   * 搜集素材
 * 黄文强 PB18061271
   * 调试代码
   * 调研整理

问题描述
--
* 初衷和动机<br>
  现代社会的发展日新月异，经济快速发展，城市面貌也飞速变化，如果能把以前的城市建筑，恢复到现在的城市中，就能有一种“昨日再现”的感觉，这就是我们选择“A Look Into the Past”的动机，新旧交替的视觉冲击会有不一样的体验。
* 创意描述<br>
  我们使用了莫斯科和国会大厦作为素材，这两对照片都是在同一角度对同一建筑物拍摄的，我们实现了把以前的照片嵌入现在的照片中，能看出面貌的巨大变化。
* 抽象模型<br>
  这个过程可以抽象成获取目标区域和图像拼接。

原理分析
--
* 特征点捕捉 (Interest Point Detection)
  拍摄两张场景有重合的照片，建立了一个图像金字塔，在不同尺度寻找Harris关键点，并根据它捕捉特征点。
* 关键点匹配
  从两幅图片的特征点中筛选出配对的点，每次选择欧氏距离最小的一对点进行配对，关键点的匹配使用Random Sample Consensus (RANSAC) 算法。以一幅图像为基准，每次从中随机选择8个点，在另一幅图像中找出配对的8个点。用8对点计算得到一个homography，将基准图中剩余的特征点按照homography变换投影到另一幅图像，统计配对点的个数。重复上述步骤2000次，得到准确配对最多的一个homography。
* 新图像的合成
  在做图像投影前，要先新建一个空白画布。比较投影后两幅图像的2维坐标的上下左右边界，选取各个方向边界的最大值作为新图像的尺寸。同时，计算得到两幅图像的交叉区域。在两幅图像的交叉区域，按照cross dissolve的方法制作两块如图6所示的蒙版，3个通道的像素值再次区间内递减（递升）。


代码实现
--

效果展示
--

下左图是现在的莫斯科，下右图是20年前的莫斯科。<br>
![现在的莫斯科](https://github.com/USTC-Computer-Vision-2021/project-cv-g-h/blob/main/Now.png)
![20年前的莫斯科](https://github.com/USTC-Computer-Vision-2021/project-cv-g-h/blob/main/Past.png)<br>
下图是我们的成果，把20年前的建筑，放入了现在的城市大背景中。<br>
![效果](https://github.com/USTC-Computer-Vision-2021/project-cv-g-h/blob/main/final_result.png)

工程结构
--

运行说明
--
```python
pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
pip install numpy
python main.py
```
