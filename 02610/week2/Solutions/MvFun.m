function [f, df, d2f] = MvFun(x)

    f = 0.5*x(1)^2 + 5*x(2)^2;
    df = [x(1); 10*x(2)];
    d2f = [1, 0; 0, 10];

