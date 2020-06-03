function I2 = mopen(I,se)
% open operation to an image
% input: I: image se: structuring elements
% output: Image
temp = imerode(I,se);
I2 = imdilate(temp,se);
end