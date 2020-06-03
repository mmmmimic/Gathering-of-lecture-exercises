function brd = doDP(Ns) 
% bdr = doDP(Ns)
% Use dynamic programming to find optimal path from top to bottom through
% an image (optimal: max cost)
%
% Ns: matrix of static costs of nodes of a graph N (pixels) - EDGES
% brd: located border

acc = zeros(size(Ns)); % cumulated cost matrix
trackback = zeros(size(Ns)); % for back tracking
acc(1,:)  = Ns(1,:); % first row of cumulated cost is first row of static cost
nAng = size(Ns,1);


maxVal = max(max(Ns)) * nAng;


% FORWARD
for I = 2:size(Ns,1)
    [acc(I,:) tb] = min(ones(3,1)*Ns(I,:) + ...
                         [ [maxVal, acc(I-1,1:end-1)] ; acc(I-1,:) ; [acc(I-1,2:end), maxVal] ]);
     trackback(I,:) = (1:size(acc,2)) + tb-2; %index hack: tb-2 = 1,2,3 -> -1,0,1 
end

brd = zeros(nAng,1); 
[m, t] = min(acc(end,:)); % the best path starts in the node with minimum 
                         % cumulated cost in the last row
brd(nAng) = t;

% BACKWARD
for I=1:nAng-1    
    brd(nAng-I) = trackback(nAng-I+1,brd(nAng-I+1));
end
