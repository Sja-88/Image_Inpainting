% XuSparse_inpainting.m
% 
% The MATLAB implementation of inpainting algorithm by Xu (2010)
%
% Inputs: 
%   - origImg        original image or corrupted image
%   - mask           implies target region (1 denotes target region) 
%   - psz:           patch size (odd scalar). If psz=5, patch size is 5x5.
%   - npsz           neighborhood patch size (odd scalar).
%   - N              patch candidate set elements number
%   - errortolerance 
%   - belta
%
% Outputs:
%   - inpaintedImg   The inpainted image; an MxNx3 matrix of doubles. 
%   - C              MxN matrix of confidence values accumulated over all iterations.
%   - D              MxN matrix of data term values accumulated over all iterations.
%   - fillMovie      A Matlab movie struct depicting the fill region over time. 
%
% 
% Author @Sijia
%   
%   
%
function [inpaintedImg,C,D,fillMovie] = XuSparse_inpainting(origImg,mask,psz,npsz,N,errortolerance,belta)

%% error check
if ~ismatrix(mask); error('Invalid mask'); end
if sum(sum(mask~=0 & mask~=1))>0; error('Invalid mask'); end
if mod(psz,2)==0; error('Patch size psz must be odd.'); end
if mod(npsz,2)==0; error('Patch size npsz must be odd.'); end


fillRegion = mask;

origImg = double(origImg);
img = origImg;
ind = img2ind(img);
origsz = [size(img,1) size(img,2)];
sourceRegion = ~fillRegion;

RefillRegion = logical(addrowscols(origsz,psz,fillRegion)); %增加黑色边框，作为已知区域处理
ResourceRegion = logical(addrowscols(origsz,psz,sourceRegion));%增加黑色边框，作为未知区域处理
Reimg = addrowscols_3(origsz,psz,img);
ReorigImg = addrowscols_3(origsz,psz,origImg);
sz = [size(Reimg,1),size(Reimg,2)];
Reind = img2ind(Reimg);




% Initialize isophote values
%[Ix(:,:,3), Iy(:,:,3)] = gradient(img(:,:,3));
%[Ix(:,:,2), Iy(:,:,2)] = gradient(img(:,:,2));
%[Ix(:,:,1), Iy(:,:,1)] = gradient(img(:,:,1));
%Ix = sum(Ix,3)/(3*255); Iy = sum(Iy,3)/(3*255);
%temp = Ix; Ix = -Iy; Iy = temp;  % Rotate gradient 90 degrees

% Initialize confidence and data terms
C = double(ResourceRegion);
D = zeros(sz);
iter = 1;
 %Visualization stuff
if nargout==4
  fillMovie(1).cdata=uint8(img); 
  fillMovie(1).colormap=[];
  iter = 2;
end

% Seed 'rand' for reproducible results (good for testing)
rand('state',0);

% Loop until entire fill region has been covered
count = 0;
while any(RefillRegion(:))
    
  % Find contour & normalized gradients of fill region
  fillRegionD = double(RefillRegion); % Marcel 11/30/05
  dR = find(conv2(fillRegionD,[1,1,1;1,-8,1;1,1,1],'same')>0);
  
     %[Nx,Ny] = gradient(double(~fillRegion)); % Marcel 11/30/05
     %[Nx,Ny] = gradient(~fillRegion);         % Original
     % N = [Nx(dR(:)) Ny(dR(:))];
     % N = normr(N);  
     % N(~isfinite(N))=0; % handle NaN and Inf
  
  
  ldR = size(dR);
  WW = zeros(ldR(1),npsz*npsz);%用于存放每个fillfront块与每个领域块匹配的权重
  DD = zeros(ldR(1),npsz*npsz);%用于存放每个fillfront块与领域块之间的SSD
  struct_sparse = zeros(sz);
  indk = 0;
  
  for k = dR'
      indk = indk+1;
      [Hp,r1,c1] = getpatch_1(sz,k,psz);%获得
      if(r1>0)
         Hnp = getpatch_2(sz,k,npsz);
         npsz1=size(Hnp);
         %nsz =size(Hnp);
         %Zp = 0;
         q = Hp(~(RefillRegion(Hp))); % fillRegionの中でパッチの部分だけ取り出して、
         C(k) = sum(C(q))/numel(Hp);% Compute confidences along the fill front
%         countp = 0;
%         for j = reshape(Hnp',[1,nsz(1)*nsz(2)])
%             [Hpj,r2]=getpatch_1(sz,j,psz);%取出对应的领域块
%             if(r2>0)
%                 countp = countp + 1;
%                 if(RefillRegion(Hpj)==zeros(psz))%若领域块完全包含在已知区域
%                    DD(k,j) = distance_cal(sz,Reimg,Hp,Hpj,RefillRegion);%求这个领域块与目标块的SSD距离
%                    WW(k,j) = exp(-DD(k,j)/25);
%                    Zp = Zp + WW(k,j);
%                 end
%             end
%         end
%         WW(k,:) = WW(k,:)/Zp;%使目标块的所有领域块权重之和为1
%         struct_sparse(k) = norm(WW(k,:))*sqrt(countp/numel(Hnp));
          toFill = RefillRegion(Hp);
          Hnpimg=ind2img(Hnp',Reimg);
          nsourceRegion=ResourceRegion(Hnp');
          [D1,Hpj,dd]=neighbordssd(Hnpimg,Reimg(r1,c1,:),toFill',nsourceRegion,sz(1));
          DD(indk,1:dd)=D1;
          WW(indk,1:dd)=exp(-sqrt(DD(indk,1:dd))/25);
          WW(indk,1:dd)=WW(indk,1:dd)/sum(WW(indk,:));
          struct_sparse(k) = norm(WW(indk,:),2)*sqrt(dd/((npsz1(1)-(psz-1)/2)*(npsz1(2)-(psz-1)/2)));
      end
  end
  
  M = max(max(struct_sparse));
  m = min(struct_sparse(struct_sparse>0));
  if(M > m)
      D(dR) = (struct_sparse(dR) - m)*0.8/(M - m)+0.2;%线性变化把结构稀疏度放缩至[0.2,1]
  end



  

  
  % Compute patch priorities = confidence term * data term
  % D(dR) = abs(Ix(dR).*N(:,1)+Iy(dR).*N(:,2)) + 0.001;
  priorities = C(dR).* D(dR);
  
  % Find patch with maximum priority, Hp
  [~,ndx] = max(priorities(:));
  p = dR(ndx(1));
  [Hp,rows,cols] = getpatch_1(sz,p,psz);
  
  if rows>0
      count = count + 1;
      toFill = RefillRegion(Hp);
      % Find N exemplars that minimizes error, Hq
      Hq = bestNexemplar(Reimg,Reimg(rows,cols,:),toFill',ResourceRegion,N);

      % Update fill region
      toFill = logical(toFill);                 % Marcel 11/30/05
      RefillRegion(Hp(toFill)) = false;

      % Propagate confidence & isophote values
      C(Hp(toFill))  = C(p);
      %Ix(Hp(toFill)) = Ix(Hq(toFill));
      %Iy(Hp(toFill)) = Iy(Hq(toFill));

      % Copy image data from Hq to Hp
      %Reind(Hp(toFill)) = Reind(Hq(toFill));
      %Reimg(rows,cols,:) = ind2img(Reind(rows,cols),ReorigImg);
      %approxpatch = zeros(psz,psz,3);%这里还有问题，要使非fillregion的像素部分不改变。3月10已修改
      %for i=1:N
      %    approxpatch=approxpatch+ind2img(Hq(:,:,i)',ReorigImg);
      %end
      [~,approxpatch]=l0_inpainting_1(psz,N,WW,Hpj,dd,ReorigImg,Hp,ndx(1),toFill,Hq,errortolerance,belta);

      Reimg(rows,cols,:)=approxpatch;
      ReorigImg(Hp(toFill))=Reimg(Hp(toFill));
      ReorigImg(Hp(toFill)+sz(1)*sz(2))=Reimg(Hp(toFill)+sz(1)*sz(2));
      ReorigImg(Hp(toFill)+2*sz(1)*sz(2))=Reimg(Hp(toFill)+2*sz(1)*sz(2));
      Reimg=ReorigImg;
      ResourceRegion(Hp(toFill))=true;
  end
  
  % Visualization stuff 

  if nargout==4
    fillRegion = recover_1(origsz,psz,RefillRegion);
    origImg = recover_3(origsz,psz,ReorigImg);
    origImg(1,1,:) = [0, 255, 0];
    ind2 = ind;
    ind2(logical(fillRegion)) = 1;
    fillMovie(iter).cdata=uint8(ind2img(ind2,origImg)); 
    fillMovie(iter).colormap=[];
  end
  iter = iter+1;

end

inpaintedImg = recover_3(origsz,psz,Reimg);
C = recover_1(origsz,psz,C);
D = recover_1(origsz,psz,D);

end
%---------------------------------------------------------------------
% Scans over the entire image (with a sliding window)
% for the exemplar with the lowest N error. Calls a MEX function.
%---------------------------------------------------------------------
function Hq = bestNexemplar(img,Ip,toFill,sourceRegion,N)
m=size(Ip,1); mm=size(img,1); n=size(Ip,2); nn=size(img,2);
sourceRegion=logical(sourceRegion);
[d,x,y,dd] = searchexemplarhelper(mm,nn,m,n,img,Ip,toFill,sourceRegion);
d=sqrt(d(1:dd));
[indexx,indexy] = sortdssd(d,x,y);
Hq = zeros(m,n,N);
for i=1:N
    Hq(:,:,i)=sub2ndx(indexx(i):m+indexx(i)-1,(indexy(i):n+indexy(i)-1)',mm);
end
end

function [D,Hpj,dd]=neighbordssd(img,Ip,toFill,sourceRegion,nTotalRows)
m=size(Ip,1); mm=size(img,1); n=size(Ip,2); nn=size(img,2);
w=(m-1)/2;
sourceRegion=logical(sourceRegion);
[D,x,y,dd] = searchexemplarhelper(mm,nn,m,n,img,Ip,toFill,sourceRegion);
D=D(1:dd);
x=x(1:dd);
y=y(1:dd);
Hpj = x+w+(y+w-1)*nTotalRows;
end
%---------------------------------------------------------------------
% Returns the indices for a 9x9 patch centered at pixel p.
%---------------------------------------------------------------------
function [Hp,rows,cols] = getpatch_1(sz,p,psz)
% [x,y] = ind2sub(sz,p);  % 2*w+1 == the patch size
w=(psz-1)/2; p=p-1; y=floor(p/sz(1))+1; p=rem(p,sz(1)); x=floor(p)+1;
if x-w  >0 && x+w < sz(1) && y-w > 0 && y+w <sz(2)
    rows = x-w:x+w;
    cols = (y-w:y+w)';
    Hp = sub2ndx(rows,cols,sz(1));
else
    rows = 0;
    cols = 0;
    Hp = 0;
end
end

%---------------------------------------------------------------------
% Returns the indices for a neighbor patch around pixel p.
%---------------------------------------------------------------------
function [Hp,rows,cols] = getpatch_2(sz,p,psz)
% [x,y] = ind2sub(sz,p);  % 2*w+1 == the patch size
w=(psz-1)/2; p=p-1; y=floor(p/sz(1))+1; p=rem(p,sz(1)); x=floor(p)+1;
rows = max(x-w,1):min(x+w,sz(1));
cols = (max(y-w,1):min(y+w,sz(2)))';
Hp = sub2ndx(rows,cols,sz(1));
end

%---------------------------------------------------------------------
% Converts the (rows,cols) subscript-style indices to Matlab index-style
% indices.  Unforunately, 'sub2ind' cannot be used for this.
%---------------------------------------------------------------------
function N = sub2ndx(rows,cols,nTotalRows)
X = rows(ones(length(cols),1),:);
Y = cols(:,ones(1,length(rows)));
N = X+(Y-1)*nTotalRows;
end

%---------------------------------------------------------------------
% Converts an indexed image into an RGB image, using 'img' as a colormap
%---------------------------------------------------------------------
function img2 = ind2img(ind,img)
for i=3:-1:1, temp=img(:,:,i); img2(:,:,i)=temp(ind); end
end


%---------------------------------------------------------------------
% Converts an RGB image into a indexed image, using the image itself as
% the colormap.
%---------------------------------------------------------------------
function ind = img2ind(img)
s=size(img); ind=reshape(1:s(1)*s(2),s(1),s(2));
end

%---------------------------------------------------------------------
% Returns the SSD between patch centered at pixel p and patch centered at pj.
%---------------------------------------------------------------------
%function DD = distance_cal(sz,masked_img,Hp,Hpj,fillregion)
%source=double(masked_img);
%x=find(fillregion(Hp)==0&fillregion(Hpj)==0);
%q=Hp(x);
%qj=Hpj(x);
%D1=sum((source(q)-source(qj)).^2);
%D2=sum((source(q+sz(1)*sz(2))-source(qj+sz(1)*sz(2))).^2);
%D3=sum((source(q+2*sz(1)*sz(2))-source(qj+2*sz(1)*sz(2))).^2);
%DD=D1+D2+D3;
%end

function reImg = addrowscols(sz,psz,img)
w = (psz-1)/2;
s1 = sz(1)+4*w;
s2 = sz(2)+4*w;
reImg = zeros(s1,s2);
reImg(2*w+1:2*w+sz(1),2*w+1:2*w+sz(2)) = img;
end

function Img = recover_1(sz,psz,ReImg)
w = (psz-1)/2;
Img(:,:) = ReImg(2*w+1:2*w+sz(1),2*w+1:2*w+sz(2));
end

function reImg = addrowscols_3(sz,psz,img)
w = (psz-1)/2;
s1 = sz(1)+4*w;
s2 = sz(2)+4*w;
reImg = 255*ones([s1,s2,3]);
%reImg(1:2*w,2*w+1:2*w+sz(2),1) = img(1:2*w,:,1);
%reImg(1:2*w,2*w+1:2*w+sz(2),2) = img(1:2*w,:,2);
%reImg(1:2*w,2*w+1:2*w+sz(2),3) = img(1:2*w,:,3);
%reImg(2*w+1+sz(1):sz(1)+4*w,2*w+1:2*w+sz(2),1) = img(sz(1)-2*w+1:sz(1),:,1);
%reImg(2*w+1+sz(1):sz(1)+4*w,2*w+1:2*w+sz(2),2) = img(sz(1)-2*w+1:sz(1),:,2);
%reImg(2*w+1+sz(1):sz(1)+4*w,2*w+1:2*w+sz(2),3) = img(sz(1)-2*w+1:sz(1),:,3);
%reImg(2*w+1:2*w+sz(1),1:2*w,1) = img(:,1:2*w,1);
%reImg(2*w+1:2*w+sz(1),1:2*w,2) = img(:,1:2*w,2);
%reImg(2*w+1:2*w+sz(1),1:2*w,3) = img(:,1:2*w,3);
%reImg(2*w+1:2*w+sz(1),2*w+1+sz(2):sz(2)+4*w,1) = img(:,sz(2)-2*w+1:sz(2),1);
%reImg(2*w+1:2*w+sz(1),2*w+1+sz(2):sz(2)+4*w,2) = img(:,sz(2)-2*w+1:sz(2),2);
%reImg(2*w+1:2*w+sz(1),2*w+1+sz(2):sz(2)+4*w,3) = img(:,sz(2)-2*w+1:sz(2),3);
reImg(2*w+1:2*w+sz(1),2*w+1:2*w+sz(2),1) = img(:,:,1);
reImg(2*w+1:2*w+sz(1),2*w+1:2*w+sz(2),2) = img(:,:,2);
reImg(2*w+1:2*w+sz(1),2*w+1:2*w+sz(2),3) = img(:,:,3);
%p1 = zeros(2*w);
%q1 = zeros(1,2*w);
%p2 = zeros(2*w);
%q2 = zeros(1,2*w);
%p3 = zeros(2*w);
%q3 = zeros(1,2*w);
%p4 = zeros(2*w);
%q4 = zeros(1,2*w);
%for j = 1:2*w
%    q1 = (j-1)*(sz(1)+4*w)+1:(j-1)*(sz(1)+4*w)+2*w;
%    p1(:,j) = q1';
%    q2 = (sz(2)+2*w+j-1)*(sz(1)+4*w)+1:(sz(2)+2*w+j-1)*(sz(1)+4*w)+2*w;
%    p2(:,j) = q2';
%    q3 = (j-1)*(sz(1)+4*w)+sz(1)+2*w+1:(j-1)*(sz(1)+4*w)+sz(1)+4*w;
%    p3(:,j) = q3';
%    q4 = (sz(2)+2*w+j-1)*(sz(1)+4*w)+sz(1)+2*w+1:(sz(2)+2*w+j-1)*(sz(1)+4*w)+sz(1)+4*w;
%    p4(:,j) = q4';
%end

%reImg(p1) = (reImg(p1+2*w*(sz(1)+4*w))+reImg(p1+2*w))/2;
%reImg(p2) = (reImg(p2-2*w*(sz(1)+4*w))+reImg(p2+2*w))/2;
%reImg(p3) = (reImg(p3+2*w*(sz(1)+4*w))+reImg(p3-2*w))/2;
%reImg(p4) = (reImg(p4-2*w*(sz(1)+4*w))+reImg(p4-2*w))/2;
end





function Img = recover_3(sz,psz,reImg)
w = (psz-1)/2;
Img(:,:,1) = reImg(2*w+1:2*w+sz(1),2*w+1:2*w+sz(2),1);
Img(:,:,2) = reImg(2*w+1:2*w+sz(1),2*w+1:2*w+sz(2),2);
Img(:,:,3) = reImg(2*w+1:2*w+sz(1),2*w+1:2*w+sz(2),3);
end


function [xx,yy]=sortdssd(d,x,y)
[~,index]=sort(d);
xx=x(index);
yy=y(index);

end

  