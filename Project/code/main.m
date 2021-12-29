clc;
clear;
close all;
%%%     里面随机算法可能导致结果不稳定，多运行几次就行。   %%%
%%%     里面随机算法可能导致结果不稳定，多运行几次就行。   %%%
%%%     里面随机算法可能导致结果不稳定，多运行几次就行。   %%%


im1='../input/1.jpg';%旧图
im2='../input/2.jpg';%新图
image1=imread(im1);
image2=imread(im2);
[row1,col1,g1]=size(image1);
[row2,col2,g2]=size(image2);

if(g1~=3)
    a=zeros(row1,col1,3);
    for i=1:3
        a(:,:,i)=image1;
    end
else
    a=image1;
end
if(g2~=3)
    b=zeros(row2,col2,3);
    for i=1:3
        b(:,:,i)=image2;
    end
else
    b=image2;
end
% a=zeros(max(row1,row2),max(col1,col2),3);
% b=zeros(max(row1,row2),max(col1,col2),3);
% if(g1==3)
%     for i=1:g1
%     a(1:row1,1:col1,i)=image1(:,:,i);
%     end
% else
%     for i=1:3
%     a(1:row1,1:col1,i)=image1;
%     end
% end
% if(g2==3)
%     for i=1:g2
%     b(1:row2,1:col2,i)=image2(:,:,i);
%     end
% else
%     for i=1:3
%     b(1:row2,1:col2,i)=image2;
%     end
% end
    
subplot(221);
imshow(uint8(a));%旧图
subplot(222);
imshow(uint8(b));%新图
[new,old]=image_stitching(uint8(a),uint8(b));
subplot(223);
imshow(mat2gray(new));%不完美拼接
subplot(224);
imshow(mat2gray(old));%旧图的变换

final_result=zeros(size(new));
area=rgb2gray(old);
old=double(old);
b=double(b);
for i=1:3
    final_result(:,:,i)=area.*new(:,:,i)+(1-area).*b(:,:,i);
end
figure;
imshow(mat2gray(final_result));
imwrite(mat2gray(final_result),'../output/result.jpg','jpg');
