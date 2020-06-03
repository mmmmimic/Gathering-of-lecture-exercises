function [ix,iy]=dpConvertPolar(nAng, lCoords, Brd)

for L = 1:nAng;
    a = Brd(L);
    b = lCoords(L,a,:);
    ix(L) = b(:,:,1);
    iy(L) = b(:,:,2);
end
