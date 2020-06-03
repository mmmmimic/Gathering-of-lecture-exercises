function p_t = PixelSizeOnCCD(h, g)
% p_t: CCD tall(in pixel)
% g: object distance
%h: object height
p_t = RealSizeOnCCD(g,h)/0.01;
end