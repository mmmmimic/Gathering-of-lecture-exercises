function Io = gamma_mapping(I,gamma)
%% gamma transformation
Itemp = double(I)./255;
Itemp = (Itemp).^gamma;
Itemp = Itemp.*255;
Io = uint8(Itemp);
end