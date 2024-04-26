% the_tensor_inpainting.m
% 
% The MATLAB implementation of inpainting algorithm by Meur et al. (2011)
%
% Inputs: 
%   - origImg        original image or corrupted image
%   - mask           implies target region (1 denotes target region) 
%   - psz:           patch size (odd scalar). If psz=5, patch size is 5x5.
%
% Outputs:
%   - inpaintedImg   The inpainted image; an MxNx3 matrix of doubles. 
%   - C              MxN matrix of confidence values accumulated over all iterations.
%   - D              MxN matrix of data term values accumulated over all iterations.
%   - fillMovie      A Matlab movie struct depicting the fill region over time. 
%
% 
%   
%   
%   
%
function [inpaintedImg,C,D,fillMovie] = the_tensor_inpainting(origImg,mask,psz)

%% error check
if ~ismatrix(mask); error('Invalid mask'); end
if sum(sum(mask~=0 & mask~=1))>0; error('Invalid mask'); end
if mod(psz,2)==0; error('Patch size psz must be odd.'); end

fillRegion = mask;

origImg = double(origImg);
img = origImg;
ind = img2ind(img);
sz = [size(img,1) size(img,2)];
sourceRegion = ~fillRegion;

% Initialize isophote values
[Ix(:,:,3), Iy(:,:,3)] = gradient(img(:,:,3));
[Ix(:,:,2), Iy(:,:,2)] = gradient(img(:,:,2));
[Ix(:,:,1), Iy(:,:,1)] = gradient(img(:,:,1));
%Ix = sum(Ix,3)/(3*255); Iy = sum(Iy,3)/(3*255);




%temp = Ix; Ix = -Iy; Iy = temp;  % Rotate gradient 90 degrees

% Initialize confidence and data terms
C = double(sourceRegion);
D = repmat(-.1,sz);
Lambda = zeros(sz);
iter = 1;
% Visualization stuff
if nargout==4
  fillMovie(1).cdata=uint8(img); 
  fillMovie(1).colormap=[];
  origImg(1,1,:) = [0, 255, 0];
  iter = 2;
end

% Seed 'rand' for reproducible results (good for testing)
rand('state',0);

% Loop until entire fill region has been covered
while any(fillRegion(:))
  % Find contour & normalized gradients of fill region
  fillRegionD = double(fillRegion); % Marcel 11/30/05
  dR = find(conv2(fillRegionD,[1,1,1;1,-8,1;1,1,1],'same')>0);
  
 % [Nx,Ny] = gradient(double(~fillRegion)); % Marcel 11/30/05
 %[Nx,Ny] = gradient(~fillRegion);         % Original
 % N = [Nx(dR(:)) Ny(dR(:))];
 % N = normr(N);  
 % N(~isfinite(N))=0; % handle NaN and Inf
 g11 = Ix.*Ix/(255)^2;
 g12 = Ix.*Iy/(255)^2;
 g22 = Iy.*Iy/(255)^2;
 g11 = sum(g11,3);
 g12 = sum(g12,3);
 g22 = sum(g22,3);
  
  % Compute confidences along the fill front
  for k=dR'
    Hp = getpatch(sz,k,psz);
    q = Hp(~(fillRegion(Hp))); % fillRegionの中でパッチの部分だけ取り出して、
    C(k) = sum(C(q))/numel(Hp);
    toFill = logical(fillRegion(Hp));
    lambda = structure_tensor(g11(Hp),g12(Hp),g22(Hp),psz,toFill);
    Lambda(k)=(lambda(2)-lambda(1))^2;
  end
  
  % Compute patch priorities = confidence term * data term
  %D(dR) = abs(Ix(dR).*N(:,1)+Iy(dR).*N(:,2)) + 0.001;
  D(dR) =0.01+(1-0.01).*exp(-8./Lambda(dR));
  priorities = C(dR).* D(dR);
  
  % Find patch with maximum priority, Hp
  [~,ndx] = max(priorities(:));
  p = dR(ndx(1));
  [Hp,rows,cols] = getpatch(sz,p,psz);
  toFill = fillRegion(Hp);
  
  % Find exemplar that minimizes error, Hq
  Hq = bestexemplar(img,img(rows,cols,:),toFill',sourceRegion);
  
  % Update fill region
  toFill = logical(toFill);                 % Marcel 11/30/05
  fillRegion(Hp(toFill)) = false;
  
  % Propagate confidence & isophote values
  C(Hp(toFill))  = C(p);
  Ix(Hp(toFill)) = Ix(Hq(toFill));
  Ix(Hp(toFill)+sz(1)*sz(2)) = Ix(Hq(toFill)+sz(1)*sz(2));
  Ix(Hp(toFill)+2*sz(1)*sz(2)) = Ix(Hq(toFill)+2*sz(1)*sz(2));
  Iy(Hp(toFill)) = Iy(Hq(toFill));
  Iy(Hp(toFill)+sz(1)*sz(2)) = Iy(Hq(toFill)+sz(1)*sz(2));
  Iy(Hp(toFill)+2*sz(1)*sz(2)) = Iy(Hq(toFill)+2*sz(1)*sz(2));
  
  % Copy image data from Hq to Hp
  ind(Hp(toFill)) = ind(Hq(toFill));
  img(rows,cols,:) = ind2img(ind(rows,cols),origImg);  

  % Visualization stuff
  if nargout==4
    ind2 = ind;
    ind2(logical(fillRegion)) = 1;
    fillMovie(iter).cdata=uint8(ind2img(ind2,origImg)); 
    fillMovie(iter).colormap=[];
  end
  iter = iter+1;
end

inpaintedImg = img;

end
%---------------------------------------------------------------------
% Scans over the entire image (with a sliding window)
% for the exemplar with the lowest error. Calls a MEX function.
%---------------------------------------------------------------------
function Hq = bestexemplar(img,Ip,toFill,sourceRegion)
m=size(Ip,1); mm=size(img,1); n=size(Ip,2); nn=size(img,2);
best = bestexemplarhelper(mm,nn,m,n,img,Ip,toFill,sourceRegion);
Hq = sub2ndx(best(1):best(2),(best(3):best(4))',mm);
end

%---------------------------------------------------------------------
% Returns the indices for a 9x9 patch centered at pixel p.
%---------------------------------------------------------------------
function [Hp,rows,cols] = getpatch(sz,p,psz)
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
% Calculate the structure tensor.
%
%---------------------------------------------------------------------
function [lambda] = structure_tensor(g11,g12,g22,psz,toFill)
sz=size(toFill);

if(sz(1)*sz(2)<psz^2)
    tempt=zeros(psz);
    tempt(1:sz(1)*sz(2))=toFill;
    toFill=tempt;
    tempt1=zeros(psz);
    tempt1(1:sz(1)*sz(2))=g11;
    g11=tempt1;
    tempt2=zeros(psz);
    tempt2(1:sz(1)*sz(2))=g12;
    g12=tempt2;
    tempt3=zeros(psz);
    tempt3(1:sz(1)*sz(2))=g22;
    g22=tempt3;
end
w=(psz-1)/2;
structure_t = zeros(2);
W = zeros(psz);
for i = 1:psz
    for j = 1:psz
        if(~toFill(i,j))
            W(i,j) = exp(-((i-w-1)^2+(j-w-1)^2)/2);
            structure_t = structure_t + W(i,j)*[g11(i,j),g12(i,j);g12(i,j),g22(i,j)];
        end
    end
end
structure_t(~isfinite(structure_t))=0;
[~,e]=eig(structure_t);
lambda = sort(diag(e));

end

