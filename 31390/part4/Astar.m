function [route, logs, dist] = Astar(start_node, end_node, map, G, D)
% Breadth First Search
%//////////////////////////////////////////////////////////////////////////
% (int) start_node
% (int) end_node
% (mtx) map
% (mtx) G
% (mtx) D
%**************************************************************************
start_node = start_node+1;
end_node = end_node+1;
% flag is a matrix to store the status of the nodes, if it has been visited
% before, set the flag to be 1
flag = zeros(size(map, 1), 1);
% create a queue to store the route
s = stack();
s = s.push(start_node);
% logs to store the searching history
logs = [];
% parents for nodes
P = flag;
% distance vector
dist = Inf(1, size(map, 1));
dist(1) = 0;
while ~isempty(s.data)
    % pop the frontier
    [front, s] = s.pop();
    flag(front) = 1;
    if front==end_node
        logs = [logs, end_node-1];
    end
    logs = [logs, front-1];
    % look for the connected nodes
    [nodes, dist, P] = getNext(front, map, flag, D, dist, P, G);
    % stack the nodes
    s = s.push(nodes);
end
% find route
route = findRoute(start_node, end_node, P);
route = route-1;
end
function [nodes, dist, P] = getNext(front, map, flag, D, dist, P, G)
    nodes = [];
    for i = 1:size(map, 1)
        if ~flag(i) && map(front, i)
            nodes = [nodes, i];
            d = dist(front)+D(front, i)+G(i);
            if d<dist(i)
               dist(i) = d; 
               P(i) = front;
            end
        end
    end
    % sort the nodes by distance
    d = dist(nodes);
    [~, idx] = sort(d, 'descend');
    nodes = nodes(idx);
end
function route = findRoute(start_node, end_node, P)
route = [];
front = end_node;
while front~=start_node
   route = [front, route];
   front = P(front);
end
route = [front, route];
end