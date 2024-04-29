clear
origimg=imread('testimage/objectremoval/a1.bmp');
masked_img=imread('testimage/objectremoval/a1_masked.bmp');
A=double(imread('testimage/objectremoval/a1_mask.bmp'));

n=find(A<255);
A(n)=0;
mask=rgb2gray(uint8(~(A/255)));

psz = 13;
npsz = 65;
N = 25;
errortolerance = 25*psz^2;
belta = 0.25;


tic
[img0,CC0,DD0,fillmove0] = Criminisi_inpainting(masked_img,mask,psz);
t(1)=toc;
%[p0,ap0]=computepsnr(double(origimg),img0);

tic
[img1,CC1,DD1,fillmove1] = the_tensor_inpainting(masked_img,mask,psz);
t(2)=toc;
%[p1,ap1]=computepsnr(double(origimg),img1);

tic
[img2,CC2,DD2,fillmove2] = XuSparse_inpainting(masked_img,mask,psz,npsz,N,errortolerance,belta);
t(3)=toc;
%[p2,ap2]=computepsnr(double(origimg),img2);

tic
[img3,CC3,DD3,fillmove3] = my_tensor_inpainting(masked_img,mask,psz);
t(4)=toc;
%[p3,ap3]=computepsnr(double(origimg),img3);

tic
[img,CC,DD,fillmove] = my_inpainting(masked_img,mask,psz,npsz,N,errortolerance,belta);
t(5)=toc;
%[p,ap]=computepsnr(double(origimg),img);

sz=size(mask);
x=sz(2)/2;
y=sz(1)+40;

figure
subplot(2,4,1),imshow(origimg),title('(a)original image','position',[x,y])
subplot(2,4,2),imshow(masked_img),title('(b)to be inpainted','position',[x,y])
subplot(2,4,3),imshow(255*mask),title('(c)mask','position',[x,y])
subplot(2,4,4),imshow(uint8(img0)),title('(d)by Criminisi et al.','position',[x,y])
subplot(2,4,5),imshow(uint8(img1)),title('(e)by Meur et al.','position',[x,y])
subplot(2,4,6),imshow(uint8(img2)),title('(f)by Xu et al.','position',[x,y])
subplot(2,4,7),imshow(uint8(img3)),title('(g)by tensor-based','position',[x,y])
subplot(2,4,8),imshow(uint8(img)),title('(h)by ours','position',[x,y])
