function line = lsqLine(edges)
% line = LSQLINE(edges) extract a line described in (alpha,r)
% parameters from a set of points
x = edges(1, :);
y = edges(2, :);
sum_x = sum(x);
sum_y = sum(y);
xx = x*x';
yy = y*y';
xy = x*y';
n = size(edges, 2);
dy = 2*sum_x*sum_y-2*n*xy;
dx = sum_x^2-sum_y^2-n*xx+n*yy;
alpha = atan2(dy, dx)/2;
if alpha<-pi
    alpha=alpha+2*pi;
end
if alpha>pi
    alpha=alpha-2*pi;
end
r = abs(mean(x)*cos(alpha)+mean(y)*sin(alpha));
line = [alpha, r];
%line = [0;0];
end