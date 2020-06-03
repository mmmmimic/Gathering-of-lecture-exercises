function SpecialWarp(im)
[m,n,d] = size(im);
im = im2double(im);
imT = im;
[X,Y] = ndgrid(1:m,1:n);
Z = ones(size(X));
X = X(:);
Y = Y(:);

% Change this, to change the basis-vectors
A = [X.^0 X Y X.^2 Y.^2 X.*Y X.^3 Y.^3 X.^2.*Y X.*Y.^2];

button = 1;
imshow(imT)
while button ~= 2
    [x,y,button] = ginput(1);
    x = round(x);
    y = round(y);
    Z(y,x) = Z(y,x)-(button-2);
    b = A\Z(:);
    ZNew = reshape(A*b,m,n);
    [diffX,diffY] = gradient(ZNew);
    diffX = diffX/min(min(diffX))*100;
    diffY = diffY/min(min(diffY))*100;
    for c = 1:d
        imT(:,:,c) = interp2(reshape(X,m,n)',reshape(Y,m,n)',im(:,:,c)',reshape(X,m,n)+diffX,reshape(Y,m,n)+diffY);
    end
    imshow(imT)
end
end