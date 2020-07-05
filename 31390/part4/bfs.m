function [route, logs] = bfs(start_node, end_node, map)
% Breadth First Search
%//////////////////////////////////////////////////////////////////////////
% (int) start_node
% (int) end_node
% (mtx) map
%**************************************************************************
start_node = start_node+1;
end_node = end_node+1;
% flag is a matrix to store the status of the nodes, if it has been visited
% before, set the flag to be 1
flag = zeros(size(map, 1), 1);
% create a stack to store the route
s = queue();
s = s.push(start_node);
% logs to store the searching history
logs = [];
% fork
fork = flag;
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
    nodes = getNext(front, map, flag, nodes);
    if ~isempty(nodes)
        for i  = 1:size(nodes, 2)
            if ~fork(nodes(i))
                fork(nodes(i)) = front;
            end
        end
    end
    % stack the nodes
    s = s.push(nodes);
end
% find route
route = findRoute(start_node, end_node, fork);
route = route-1;
end
function nodes = getNext(front, map, flag, nodes)
for i = 1:size(map, 1)
    if ~flag(i) && map(front, i)
        % start by stacking the lowest of the s numbers
        nodes = [nodes, i];
        % start by stacking the highest of the s numbers
        %nodes = [i, nodes];
    end
end
end
function route = findRoute(start_node, end_node, fork)
route = [];
front = end_node;
while front~=start_node
    route = [front, route];
    front = fork(front);
end
route = [front, route];
end