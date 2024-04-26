function [e,approxpatch]=l0_inpainting(psz,N,WW,Hpj,dd,originalimg,Hp,p,toFill,Hq,errortolerance,belta)
%w = (psz-1)/2;
approxpatch = zeros(psz,psz,3);
sz=size(originalimg(:,:,1));
knowpatch = ~toFill;
unknowpatch  = toFill;
Tknowpart(:,:,1) = knowpatch.*originalimg(Hp);
Tknowpart(:,:,2) = knowpatch.*originalimg(Hp+sz(1)*sz(2));
Tknowpart(:,:,3) = knowpatch.*originalimg(Hp+2*sz(1)*sz(2));

sumT1 = zeros(psz);
sumT2 = zeros(psz);
sumT3 = zeros(psz);
for j=1:dd
    hpj = getpatch_1(sz,Hpj(j),psz);
    T1(:,:,1)=WW(p,j)*originalimg(hpj);
    sumT1 = sumT1 + T1(:,:,1);
    T1(:,:,2)=WW(p,j)*originalimg(hpj+sz(1)*sz(2));
    sumT2 = sumT2 + T1(:,:,2);
    T1(:,:,3)=WW(p,j)*originalimg(hpj+2*sz(1)*sz(2));
    sumT3 = sumT3 + T1(:,:,3);
end

Tapproxpart(:,:,1) = sqrt(belta)*unknowpatch.*sumT1;
Tapproxpart(:,:,2) = sqrt(belta)*unknowpatch.*sumT2;
Tapproxpart(:,:,3) = sqrt(belta)*unknowpatch.*sumT3;

for i = 1:3
    Dknowpart(:,:,i) = knowpatch;
    Dunknowpart(:,:,i) = unknowpatch;
end

patchT = [Tknowpart;Tapproxpart];
patchD = [Dknowpart;Dunknowpart];

patchT2V = reshape(patchT,[2*psz^2*3,1]);
patchD2V = reshape(patchD,[2*psz^2*3,1]);


n=1;
ph(:,:,1) = originalimg(Hq(:,:,1));
ph(:,:,2) = originalimg(Hq(:,:,1)+sz(1)*sz(2));
ph(:,:,3) = originalimg(Hq(:,:,1)+2*sz(1)*sz(2));
Ph = [ph;ph];
Ph2V = reshape(Ph,[2*psz^2*3,1]);
X(:,1) = patchD2V.*Ph2V;
S=zeros(N,1);
S(1) = 1;

colvec = ones(n,1);
G = (patchT2V*colvec'-X)'*(patchT2V*colvec'-X);
szg=size(G);
glambda=ones([1,szg(1)]);
glambda=diag(glambda);
alpha = (G+glambda)\colvec/(colvec'*((glambda+G)\colvec));
alpham = repmat(alpha',[2*psz^2*3,1]);
patchT2Vm = repmat(patchT2V,[1,n]);
e = norm(X.*alpham-patchT2Vm);
et = e;
contin = true;


while(contin)
    if (S(n+1)==0)
       ph(:,:,1) = originalimg(Hq(:,:,n+1));
       ph(:,:,2) = originalimg(Hq(:,:,n+1)+sz(1)*sz(2));
       ph(:,:,3) = originalimg(Hq(:,:,n+1)+2*sz(1)*sz(2));
       Ph = [ph;ph];
       Ph2V = reshape(Ph,[2*psz^2*3,1]);
       X(:,n+1) = patchD2V.*Ph2V;
    end
    colvec = ones(n+1,1);
    G = (patchT2V*colvec'-X)'*(patchT2V*colvec'-X);
    szg=size(G);
    glambda=ones([1,szg(1)]);
    glambda=diag(glambda);
    alphan = (G+glambda)\colvec/(colvec'*((G+glambda)\colvec));
    alpham = repmat(alphan',[2*psz^2*3,1]);
    patchT2Vm = repmat(patchT2V,[1,n+1]);
    e = norm(X.*alpham-patchT2Vm);
    if(n<N && e>=errortolerance && e<=et)
       n=n+1;
       S(n)=1;
       et = e;
       alpha=alphan;
    else
        contin = false;
    end

end
l=size(alpha);
numberofpatch = l(1);
for i = 1:numberofpatch
  approxpatch(:,:,1) = approxpatch(:,:,1) + alpha(i)*originalimg(Hq(:,:,i)');
  approxpatch(:,:,2) = approxpatch(:,:,2) + alpha(i)*originalimg(Hq(:,:,i)'+sz(1)*sz(2));
  approxpatch(:,:,3) = approxpatch(:,:,3) + alpha(i)*originalimg(Hq(:,:,i)'+2*sz(1)*sz(2));
end

end