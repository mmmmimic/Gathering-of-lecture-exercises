function Io = HistStretch(I)
%Input I: image
%Output Io: mapping the image to [0 255]
Itemp = double(I);
max_p = max(max(Itemp)); % max pixel
min_p = min(min(Itemp)); % min pixel
max_t = 255;% max target
min_t = 0;% mean target
It = (Itemp-min_p)*(max_t-min_t)/(max_p-min_p)+min_t;
Io = uint8(It);
end