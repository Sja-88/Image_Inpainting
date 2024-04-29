clear
origimg=imread('testimage/psnr/texttest/a05.bmp');
masked_img=imread('testimage/psnr/texttest/a05_masked.bmp');
A=double(imread('testimage/psnr/texttest/a05_mask.bmp'));

n=find(A<255);
A(n)=0;
mask=logical(~(A/255));
%mask=rgb2gray(uint8(~(A/255)));

psz = 7;
npsz = 51;
N = 25;
errortolerance = 25*psz^2;
belta = 0.25;



tic
[img1,CC1,DD1,fillmove1] = Criminisi_inpainting(masked_img,mask,psz);
t(1)=toc;
[p0,ap(1)]=computepsnr(double(origimg),img1);

tic
[img2,CC2,DD2,fillmove2] = the_tensor_inpainting(masked_img,mask,psz);
t(2)=toc;
[p1,ap(2)]=computepsnr(double(origimg),img2);

tic
[img3,CC3,DD3,fillmove3] = XuSparse_inpainting(masked_img,mask,psz,npsz,N,errortolerance,belta);
t(3)=toc;
[p2,ap(3)]=computepsnr(double(origimg),img3);

tic
[img4,CC4,DD4,fillmove4] = my_tensor_inpainting(masked_img,mask,psz);
t(4)=toc;
[p3,ap(4)]=computepsnr(double(origimg),img4);

tic
[img5,CC5,DD5,fillmove5] = my_inpainting(masked_img,mask,psz,npsz,N,errortolerance,belta);
t(5)=toc;
[p,ap(5)]=computepsnr(double(origimg),img5);


sz=size(mask);
x=sz(2)/2;
y=sz(1)+40;

figure
subplot(2,4,1),imshow(origimg),title('(a0)original image','position',[x,y])
subplot(2,4,2),imshow(masked_img),title('(b0)to be inpainted','position',[x,y])
subplot(2,4,3),imshow(255*mask),title('(c0)mask','position',[x,y])
subplot(2,4,4),imshow(uint8(img1)),title('(d0)by Criminisi et al.','position',[x,y])
subplot(2,4,5),imshow(uint8(img2)),title('(e0)by Meur et al.','position',[x,y])
subplot(2,4,6),imshow(uint8(img3)),title('(f0)by Xu et al','position',[x,y])
subplot(2,4,7),imshow(uint8(img4)),title('(g0)by tensor-based','position',[x,y])
subplot(2,4,8),imshow(uint8(img5)),title('(h0)by ours','position',[x,y])
