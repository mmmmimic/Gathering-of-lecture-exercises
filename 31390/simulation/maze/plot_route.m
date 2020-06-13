function plot_route( route, fig_num )

figure(fig_num)

if size(route,2) == 2

    for i = 1:size(route,1)-1
        hold on
        plot([route(i,1), route(i+1,1)]-1, [route(i,2), route(i+1,2)]-1, 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 3)
        hold off
    end
    
else
    route = route -1;

    for i = 2:length(route)
        plot3([route(i-1,1)+0.0,route(i,1)+0.0], ...
              [route(i-1,2)+0.0,route(i,2)+0.0], ...
              [route(i-1,3)+0.5,route(i,3)+0.5], ...
              'color',[0,0,0],'linewidth',5)
        hold on
        pause(0.1)
    end
    hold off
   
end

end