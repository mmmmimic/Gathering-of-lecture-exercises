function I2 = mclose(I,se)
% open operation to an image
% input: I: image se: structuring elements
% output: Image
temp = imdilate(I,se);
I2 = imerode(temp,se);
end