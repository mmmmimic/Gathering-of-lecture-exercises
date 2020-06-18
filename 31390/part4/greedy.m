function [route, logs] = greedy(start_node, end_node, map, G)
% Best First Search
%//////////////////////////////////////////////////////////////////////////
% (int) start_node
% (int) end_node
% (mtx) map
% (mtx) G
%**************************************************************************
start_node = start_node+1;
end_node = end_node+1;
% flag is a matrix to store the status of the nodes, if it has been visited
% before, set the flag to be 1
flag = zeros(size(map, 1), 1);
% create a stack to store the route
s = stack();
s = s.push(start_node);
% logs to store the searching history
logs = [];
% parent of the node
P = flag;
while ~isempty(s.data)
    % pop the frontier
    [front, s] = s.pop();
    flag(front) = 1;
    if front==end_node
        logs = [logs, end_node-1];
        break;
    end
    logs = [logs, front-1];
    % look for the connected nodes
    nodes = [];
    [nodes, P] = getNext(front, map, flag, nodes, G, P);
    % stack the nodes
    s = s.push(nodes);
end
% find route
route = findRoute(end_node, P);
route = route-1;
end
function [nodes, P] = getNext(front, map, flag, nodes, G, P)
    for i = 1:size(map, 1)
        if ~flag(i) && map(front, i)
            nodes = [nodes, i];
            P(nodes) = front;
        end
    end
    g = G(nodes);
    [~, idx] = sort(g, 'descend');
    nodes = nodes(idx);
end
function route = findRoute(end_node, P)
    route = [];
    front = end_node;
    while front
      route = [front, route];
      front = P(front);
    end
end
