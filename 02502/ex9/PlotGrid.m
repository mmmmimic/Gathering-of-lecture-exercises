function PlotGrid(XY)
XY = XY(1:2,:)
n = sqrt(length(XY));
XY = reshape(XY',[n n 2]);
plot(XY(:,:,1),XY(:,:,2),'r.')
hold on
for i = 1:n
    for j = 1:n-1
        plot([XY(i,j,1) XY(i,j+1,1)],[XY(i,j,2) XY(i,j+1,2)],'r')
        plot([XY(j,i,1) XY(j+1,i,1)],[XY(j,i,2) XY(j+1,i,2)],'r')
    end
end
xlim([min(min(XY(:,:,1)))-1 max(max(XY(:,:,1)))+1])
ylim([min(min(XY(:,:,2)))-1 max(max(XY(:,:,2)))+1])
axis equal
hold off;
end
