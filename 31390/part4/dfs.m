function [route, logs] = dfs(start_node, end_node, map)
% Depth First Search
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
s = stack();
s = s.push(start_node);
% initialize route
route = [];
% logs to store the searching history
logs = [];
% store forks
fork = stack();
fork = fork.push(1);
while ~isempty(s.data)
    % pop the frontier
    [front, s] = s.pop();
    flag(front) = 1;
    if front==end_node
        route = [route, end_node-1];
        logs = [logs, end_node-1];
        break;
    end
    route = [route, front-1];
    logs = [logs, front-1];
    % look for the connected nodes
    nodes = [];
    nodes = getNext(front, map, flag, nodes);
    if ~isempty(nodes) && ~sum(fork.data==front)
       fork = fork.push(front);
    end
    if isempty(nodes)
       [f, fork] = fork.pop();
       idx = find(route==(f-1));
       if isempty(getNext(f, map, flag, []))
           route(idx:end) = []; 
       else
           route(idx+1:end) = [];
       end
    end    
    % stack the nodes
    s = s.push(nodes);
end
end
function nodes = getNext(front, map, flag, nodes)
    for i = 1:size(map, 1)
        if ~flag(i) && map(front, i)
            % start by stacking the lowest of the s numbers
            %nodes = [nodes, i];
            % start by stacking the highest of the s numbers
            nodes = [i, nodes];
        end
    end
end