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
