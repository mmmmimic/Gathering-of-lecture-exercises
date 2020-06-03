function imagegrid(h,imsize)
% Call: imagegrid(h,imsize)
% h is handle to the axes. Normally just send gca
% imsize is the size of the image. Normally just send size(I)
  set(h,'xtick',1.5:imsize(1)+.5,'ytick',1.5:imsize(2)+.5,...
    'XTickLabel','',...
    'YTickLabel','',...
    'xcolor','r', 'ycolor', 'r','GridLineStyle','-')
  grid on,axis image
end