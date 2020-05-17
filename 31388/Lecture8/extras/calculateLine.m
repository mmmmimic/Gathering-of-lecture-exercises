function line = lsqLine(edges)

    ex = sum(edges(1,:));
    ey = sum(edges(2,:));
    ex2 = edges(1,:)*(edges(1,:))';
    ey2 = edges(2,:)*edges(2,:)';
    exy = edges(1,:)*edges(2,:)';
    
    [m,n] = size(edges);
    
    theta=atan2(2*ex*ey-2*n*exy,ex*ex-ey*ey-n*(ex2-ey2))/2;
	radius=(ex/n)*cos(theta)+(ey/n)*sin(theta);
    
    line = [theta;radius];
    line = correctLine(line);
end