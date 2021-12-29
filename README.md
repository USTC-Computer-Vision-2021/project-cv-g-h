基于图像拼接的A Look Into the Past
===
成员及分工
---
 * 龚愉皓 PB18061299
   * 分工
 * 黄文强 PB18061271
   * 分工

问题描述
--
* 初衷和动机<br>
  现代社会的发展日新月异，经济快速发展，城市面貌也飞速变化，如果能把以前的城市建筑，恢复到现在的城市中，就能有一种“昨日再现”的感觉，这就是我们选择“A Look Into the Past”的动机，新旧交替的视觉冲击会有不一样的体验。
* 创意描述<br>
  我们使用了20年前的莫斯科照片和现在的莫斯科照片作为素材，这两张照片是在同一角度对同一建筑物拍摄的，我们实现了把20年前的建筑物放入现在的城市背景中，能看出城市面貌的巨大变化。
* 抽象模型<br>
  这个过程可以抽象成获取目标区域和图像拼接。

原理分析
--
我们为了区分我们关注的物体与其所在背景，我们采用提取前景的方法，这个方向有很多的研究，其中一种比较简单的是，根据背景颜色的相似性，来生成mask，在生成掩膜后，由于生成的方式比较简单，因此需要进一步处理，对掩膜进行连通域分析，这可以使mask更加平滑，易于使用，最后，我们希望过去的照片做前景，现在的照片做背景，因此，我们只需要利用mask把二者拼接起来即可。

代码实现
--

效果展示
--
下左图是现在的莫斯科，下右图是20年前的莫斯科。<br>
![现在的莫斯科](https://github.com/USTC-Computer-Vision-2021/project-cv-g-h/blob/main/Now.png)
![20年前的莫斯科](https://github.com/USTC-Computer-Vision-2021/project-cv-g-h/blob/main/Past.png)
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
