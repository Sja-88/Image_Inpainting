clear

psz = 11;
npsz = 51;
N = 25;
errortolerance = 25*psz^2;
belta = 0.25;

origimg_1=imread('testimage/psnr/structtest/a01.bmp');
masked_img_1=imread('testimage/psnr/structtest/a01_masked.bmp');
A=double(imread('testimage/psnr/structtest/a01_mask.bmp'));
n=find(A<255);
A(n)=0;
mask_1=logical(~(A/255));
%tic
%[img1_1,~,~,~] = my_inpainting(masked_img_1,mask_1,psz,npsz,N,errortolerance,belta);
%t_1(1)=toc;
%[~,ap_1(1)]=computepsnr(double(origimg_1),img1_1);

tic
[img1,~,~,~] = my_inpainting_l01(masked_img_1,mask_1,psz,npsz,N,errortolerance,belta);
t_1=toc
[~,ap_1]=computepsnr(double(origimg_1),img1);

origimg_2=imread('testimage/psnr/structtest/a02.bmp');
masked_img_2=imread('testimage/psnr/structtest/a02_masked.bmp');
A=double(imread('testimage/psnr/structtest/a02_mask.bmp'));
n=find(A<255);
A(n)=0;
mask_2=logical(~(A/255));
%tic
%[img2_1,~,~,~] = my_inpainting(masked_img_2,mask_2,psz,npsz,N,errortolerance,belta);
%t_2(1)=toc;
%[~,ap_2(1)]=computepsnr(double(origimg_2),img2_1);

tic
[img2,~,~,~] = my_inpainting_l01(masked_img_2,mask_2,psz,npsz,N,errortolerance,belta);
t_2=toc
[~,ap_2]=computepsnr(double(origimg_2),img2);

origimg_3=imread('testimage/psnr/structtest/a03.bmp');
masked_img_3=imread('testimage/psnr/structtest/a03_masked.bmp');
A=double(imread('testimage/psnr/structtest/a03_mask.bmp'));
n=find(A<255);
A(n)=0;
mask_3=logical(~(A/255));
%tic
%[img3_1,~,~,~] = my_inpainting(masked_img_3,mask_3,psz,npsz,N,errortolerance,belta);
%t_3(1)=toc;
%[~,ap_3(1)]=computepsnr(double(origimg_3),img3_1);

tic
[img3,~,~,~] = my_inpainting_l01(masked_img_3,mask_3,psz,npsz,N,errortolerance,belta);
t_3=toc
[~,ap_3]=computepsnr(double(origimg_3),img3);

origimg_4=imread('testimage/psnr/structtest/a04.bmp');
masked_img_4=imread('testimage/psnr/structtest/a04_masked.bmp');
A=double(imread('testimage/psnr/structtest/a04_mask.bmp'));
n=find(A<255);
A(n)=0;
mask_4=logical(~(A/255));
%tic
%[img4_1,~,~,~] = my_inpainting(masked_img_4,mask_4,psz,npsz,N,errortolerance,belta);
%t_4(1)=toc;
%[~,ap_4(1)]=computepsnr(double(origimg_4),img4_1);

tic
[img4,~,~,~] = my_inpainting_l01(masked_img_4,mask_4,psz,npsz,N,errortolerance,belta);
t_4=toc
[~,ap_4]=computepsnr(double(origimg_4),img4);




