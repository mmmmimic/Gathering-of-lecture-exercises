function lineHandle = plotLine(lineParams,color,linestyle)
axisTemp = axis();
alpha = lineParams(1);
r = lineParams(2);

if(nargin<3)
    linestyle = '-';
end

X = r*cos(alpha) - [-1000,1000]*sin(alpha);
Y = r*sin(alpha) + [-1000,1000]*cos(alpha);
lineHandle = line(X,Y,'color', color,'linestyle',linestyle);
axis(axisTemp);
