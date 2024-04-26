function [ P, ap ] = computepsnr( i0, i )
%用来计算修复后图像的峰值信噪比
%   Detailed explanation goes here

[m,n]=size(i(:,:,1));
mse=sum(sum((i0-i).^2))/(m*n);
P=10*log10(255^2/mse);
ap=sum(P)/3;

end