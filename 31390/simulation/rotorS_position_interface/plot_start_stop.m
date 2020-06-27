function plot_start_stop(start, end_, fig_num)

offset = 0;
wallSize = 500;

figure(fig_num)

if size(start,2) == 2

    % Mark the start with green
    hold on
    scatter(start(1)-offset, start(2)-offset, [wallSize/2], [0,1,0], 'filled')
    % Mark the end with red
    scatter(end_(1)-offset, end_(2)-offset, [wallSize/2], [1,0.6,0], 'filled')
    hold off

else
   
    % Mark the start with green
    scatter3(start(1)+0.0, start(2)+0.0, start(3)+0.5, ...
             wallSize, [0,1,0],'filled')
    hold on

    % Mark the end with red
    scatter3(end_(1)+0.0, end_(2)+0.0, end_(3)+0.5, ...
             wallSize, [0,0,1], 'filled')
    hold on
    
end
    
end