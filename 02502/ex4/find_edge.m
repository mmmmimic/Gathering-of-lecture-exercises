function edge = find_edge(img,key)
if key == 0
    filter = fspecial('prewitt');
end
if key == 1
    filter = fspecial('sobel');
end
im1 = imfilter(img,filter);
im2 = imfilter(img,filter');
im3 = imfilter(img,-filter');
edge = im1+im2+im3;
end