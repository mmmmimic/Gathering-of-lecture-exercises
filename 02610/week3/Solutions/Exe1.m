close all, clear all

x = [0;-1];  %[0;0.5]; %
[f, df, d2f] = Myfun(x);

m = @(p1,p2)(f+df(1)*p1+df(2)*p2+0.5*(d2f(1,1)*p1.^2+(d2f(1,2)+d2f(2,1))*p1.*p2+d2f(2,2)*p2.^2));

p1=-3:0.05:3;
p2=-3:0.05:3;
[P1,P2]=meshgrid(p1,p2);
F=m(P1,P2);

figure,
v=[0:5:200];
[c,h]=contour(P1,P2,F,v,'linewidth',2);
colorbar, axis image,
xlabel('p_1','fontsize',14),
ylabel('p_2','fontsize',14),


%% exact solver for subproblem

p_star = -d2f\df;
norm_p_star = norm(p_star); 

min_p = [];

for delta = 0.1:0.1:2
    if norm_p_star < delta
        p = p_star;
        min_p = [min_p, p];
    else
        % in this example we only have easy case
        % we use Newton to find the root of phi_2
        lambda=20;
        R = chol(d2f+lambda*eye(size(d2f)));
        pl = -R\(R'\df);
        norm_pl = norm(pl);        
        phi2 = 1/delta-1/norm_pl;
        
        while phi2 > 1e-8 
            ql = R'\pl;
            norm_ql = norm(ql);
            lambda = lambda+(norm_pl/norm_ql)^2*((norm_pl-delta)/delta);
            
            R = chol(d2f+lambda*eye(size(d2f)));
            pl = -R\(R'\df);
            norm_pl = norm(pl);
            phi2 = 1/delta-1/norm_pl;
        end
        p = pl;
        min_p = [min_p, p];
    end
end

hold on,
plot(min_p(1,:), min_p(2, :),'ro');
hold off,
        
        