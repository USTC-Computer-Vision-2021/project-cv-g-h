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
```
function [xp, yp, value] = harris(input_image, sigma,thd, r)

g1 = fspecial('gaussian', 7, 1);
gray_image = imfilter(input_image, g1);

h = fspecial('sobel');
Ix = imfilter(gray_image,h,'replicate','same');
Iy = imfilter(gray_image,h','replicate','same');

g = fspecial('gaussian',fix(6*sigma), sigma);

Ix2 = imfilter(Ix.^2, g, 'same').*(sigma^2); 
Iy2 = imfilter(Iy.^2, g, 'same').*(sigma^2);
Ixy = imfilter(Ix.*Iy, g, 'same').*(sigma^2);

R = (Ix2.*Iy2 - Ixy.^2)./(Ix2 + Iy2 + eps); 

R([1:20, end-20:end], :) = 0;
R(:,[1:20,end-20:end]) = 0;

d = 2*r+1; 
localmax = ordfilt2(R,d^2,true(d)); 
R = R.*(and(R==localmax, R>thd));

[xp,yp,value] = find(R);
```
3.在两个图像中的每个关键点周围提取固定大小的补丁，并简单地通过将每个补丁中的像素值“展平”为一维向量来形成描述符。<br>
```
function [descriptors] = getFeatureDescriptor(input_image, xp, yp, sigma)

g = fspecial('gaussian', 5, sigma);
blurred_image = imfilter(input_image, g, 'replicate','same');

npoints = length(xp);
descriptors = zeros(npoints,64);

for i = 1:npoints
patch = blurred_image(xp(i)-20:xp(i)+19, yp(i)-20:yp(i)+19);
patch = imresize(patch, .2);
descriptors(i,:) = reshape((patch - mean2(patch))./std2(patch), 1, 64); 
end
```
4.计算一幅图像中的每个描述符与另一幅图像中的每个描述符之间的距离。<br>
```
function n2 = dist2(x, c)

[ndata, dimx] = size(x);
[ncentres, dimc] = size(c);
if dimx ~= dimc
error('Data dimension does not match dimension of centres')
end

n2 = (ones(ncentres, 1) * sum((x.^2)', 1))' + ...
ones(ndata, 1) * sum((c.^2)',1) - ...
2.*(x*(c'));

if any(any(n2<0))
n2(n2<0) = 0;
end
```
5.根据上面获得的成对描述符距离矩阵选择假定的匹配项。<br>
```
dist = dist2(des_A,des_B);
[ord_dist, index] = sort(dist, 2);
ratio = ord_dist(:,1)./ord_dist(:,2);
threshold = 0.5;
idx = ratio<threshold;

x_A = x_A(idx);
y_A = y_A(idx);
x_B = x_B(index(idx,1));
y_B = y_B(index(idx,1));
npoints = length(x_A);

matcher_A = [y_A, x_A, ones(npoints,1)]'; 
matcher_B = [y_B, x_B, ones(npoints,1)]'; 
```
6.运行 RANSAC 以估计仿射变换并将一个图像映射到另一个图像上。<br>
```
[hh, ~] = ransacfithomography(matcher_B, matcher_A, npoints, 10);

[newH, newW, newX, newY, xB, yB] = getNewSize(hh, height_wrap, width_wrap, height_unwrap, width_unwrap);

[X,Y] = meshgrid(1:width_wrap,1:height_wrap);
[XX,YY] = meshgrid(newX:newX+newW-1, newY:newY+newH-1);
AA = ones(3,newH*newW);
AA(1,:) = reshape(XX,1,newH*newW);
AA(2,:) = reshape(YY,1,newH*newW);

AA = hh*AA;
XX = reshape(AA(1,:)./AA(3,:), newH, newW);
YY = reshape(AA(2,:)./AA(3,:), newH, newW);

newImage(:,:,1) = interp2(X, Y, double(image_A(:,:,1)), XX, YY);
newImage(:,:,2) = interp2(X, Y, double(image_A(:,:,2)), XX, YY);
newImage(:,:,3) = interp2(X, Y, double(image_A(:,:,3)), XX, YY);
```
7.使用估计的变换将一个图像扭曲到另一个图像上。<br>
```
function [newImage] = blend(warped_image, unwarped_image, x, y)

warped_image(isnan(warped_image))=0;
maskA = (warped_image(:,:,1)>0 |warped_image(:,:,2)>0 | warped_image(:,:,3)>0);
newImage = zeros(size(warped_image));
newImage(y:y+size(unwarped_image,1)-1, x: x+size(unwarped_image,2)-1,:) = unwarped_image;
mask = (newImage(:,:,1)>0 | newImage(:,:,2)>0 | newImage(:,:,3)>0);
mask = and(maskA, mask);

[~,col] = find(mask);
left = min(col);
right = max(col);
mask = ones(size(mask));
if( x<2)
mask(:,left:right) = repmat(linspace(0,1,right-left+1),size(mask,1),1);
else
mask(:,left:right) = repmat(linspace(1,0,right-left+1),size(mask,1),1);
end

warped_image(:,:,1) = warped_image(:,:,1).*mask;
warped_image(:,:,2) = warped_image(:,:,2).*mask;
warped_image(:,:,3) = warped_image(:,:,3).*mask;

if( x<2)
mask(:,left:right) = repmat(linspace(1,0,right-left+1),size(mask,1),1);
else
mask(:,left:right) = repmat(linspace(0,1,right-left+1),size(mask,1),1);
end
newImage(:,:,1) = newImage(:,:,1).*mask;
newImage(:,:,2) = newImage(:,:,2).*mask;
newImage(:,:,3) = newImage(:,:,3).*mask;

c=newImage;
newImage(:,:,1) = warped_image(:,:,1) + newImage(:,:,1);
newImage(:,:,2) = warped_image(:,:,2) + newImage(:,:,2);
newImage(:,:,3) = warped_image(:,:,3) + newImage(:,:,3);
```
8.创建一个足够大的新图像以容纳全景图并将两个图像合成到其中。<br>
```
function [newH, newW, x1, y1, x2, y2] = getNewSize(transform, h2, w2, h1, w1)

[X,Y] = meshgrid(1:w2,1:h2);
AA = ones(3,h2*w2);
AA(1,:) = reshape(X,1,h2*w2);
AA(2,:) = reshape(Y,1,h2*w2);

newAA = transform\AA;
new_left = fix(min([1,min(newAA(1,:)./newAA(3,:))]));
new_right = fix(max([w1,max(newAA(1,:)./newAA(3,:))]));
new_top = fix(min([1,min(newAA(2,:)./newAA(3,:))]));
new_bottom = fix(max([h1,max(newAA(2,:)./newAA(3,:))]));

newH = new_bottom - new_top + 1;
newW = new_right - new_left + 1;
x1 = new_left;
y1 = new_top;
x2 = 2 - new_left;
y2 = 2 - new_top;
```
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
