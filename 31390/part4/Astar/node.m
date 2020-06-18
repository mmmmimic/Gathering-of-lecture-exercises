classdef node
    properties
        % Node which is this node's parent
        parent
        % The position of the node
        position
        % visitied
        v
        % The values which the node uses to calculate
        g
        h
        f
    end
    
    methods
        function h = calc_dist(obj, end_pos)
            h = sqrt((obj.position(1)-end_pos(1))^2 + ...
                     (obj.position(2)-end_pos(2))^2);
        end

        function h = calc_dist_3d(obj, end_pos)
            h = sqrt((obj.position(1)-end_pos(1))^2 + ...
                     (obj.position(2)-end_pos(2))^2 + ...
                     (obj.position(3)-end_pos(3))^2);
        end
    end
end