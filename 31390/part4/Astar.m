function [route, logs, g] = Astar(start_node, end_node, map, H, D)
%  A* Search
%//////////////////////////////////////////////////////////////////////////
% (int) start_node series number
% (int) end_node series number
% (mtx) map, adjacent matrix of the graph, 24x24
% (mtx) H: Huristic matrix, 1x24
% (mtx) D: path length matrix, 24x24, D(m, n) returns the distance
% between node m and n
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
% F to store the lowest f
F = Inf(1, size(map, 1));
F(1) = H(1)+g(1);
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
            f = g(i)+H(i);
            if f<F(i)
                F(i) = f;
                P(i) = front;
            end
        end
    end
    % sort the nodes by distance
    [~, idx] = sort(F(nodes));
    nodes = nodes(idx);
    % stack the nodes
    s = s.push(nodes);
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