function [route, logs, G] = Dijkstra(start_node, end_node, map, D)
% Dijkstra's algorithm
%//////////////////////////////////////////////////////////////////////////
% (int) start_node
% (int) end_node
% (mtx) map
% (mtx) D
%**************************************************************************
start_node = start_node+1;
end_node = end_node+1;
% flag is a matrix to store the status of the nodes, if it has been visited
% before, set the flag to be 1
flag = zeros(size(map, 1), 1);
% create a queue to store the route
s = queue();
s = s.push(start_node);
% logs to store the searching history
logs = [];
% parents for nodes
P = flag;
% distance vector
g = Inf(1, size(map, 1));
g(1) = 0;
G = [];
while ~isempty(s.data)
    % pop the frontier
    [front, s] = s.pop();
    flag(front) = 1;
    if front==end_node
        logs = [logs, end_node-1];
    end
    logs = [logs, front-1];
    % look for the connected nodes
    nodes = [];
    for i = 1:size(map, 1)
        if map(front, i) && (g(front)+D(front, i))<g(i)
            g(i) = g(front)+D(front, i);
            P(i) = front;
        end
        if ~flag(i) && map(front, i)
            nodes = [nodes, i];
        end
    end
    % sort the nodes by cost, Ascend
    [~, idx] = sort(g(nodes));
    nodes = nodes(idx);
    % stack the nodes
    s = s.push(nodes);
    G = [G;g];
end
% find route
route = findRoute(end_node, P);
route = route-1;
end
function route = findRoute(end_node, P)
route = [];
front = end_node;
while P(front)
    route = [front, route];
    front = P(front);
end
route = [front, route];
end