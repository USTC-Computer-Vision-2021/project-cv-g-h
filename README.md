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

首先，拍摄两张场景有重合的照片。为了保证有足够多的公共特征点，照片的重合度应该保证在30%以上。将两张照片转换为灰度图像，对图像做σ=1的高斯模糊。在Matthew的文章中，他建立了一个图像金字塔，在不同尺度寻找Harris关键点。考虑到将要拼接的照片视野尺寸接近，故简化此步骤，仅在原图提取特征点。

接下来用sobel算子计算图像在x、y两个方向亮度的梯度，用σ=1.5的高斯函数对梯度做平滑处理，减小噪点对亮度的影响。很容易发现，若我们求一小块区域内亮度的累加值，在图像变化平缓的区域上下左右移动窗口累加值的变化并不明显；在物体的边缘，沿着边缘方向的变化也不明显；而在关键点附近，轻微的移动窗口都会强烈改变亮度的累加值，如下图所示。
![](https://images.cnitblog.com/i/606248/201403/040127281483199.png)


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
