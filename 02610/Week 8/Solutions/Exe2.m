H = [6, 2; 2, 10];
g = [7; 3];
A = [3, 1];
b = -5;

x_unc = quadprog(H,g)
x_con = quadprog(H,g,A,b)