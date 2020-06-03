function Io = ImgThreshold(Img, T)
Img = double(Img)./255;
[row, col] = size(Img);
Io = ones(row,col);
Io = uint8(Io);
for i = 1:row
    for j = 1:col
        if Img(i,j)<T
            Io(i,j) = uint8(0);
        end
    end
end
end