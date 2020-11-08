function x = linearLSQ(A, y)

[Q, R] = qr(A,0);
x = R\(Q'*y);
