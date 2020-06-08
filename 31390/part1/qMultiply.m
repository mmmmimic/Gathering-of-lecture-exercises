function q = qMultiply(q1,q2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
a1 = q1(1);
b1 = q1(2);
c1 = q1(3);
d1 = q1(4);
a2 = q2(1);
b2 = q2(2);
c2 = q2(3);
d2 = q2(4);
q = [a1*a2-b1*b2-c1*c2-d1*d2;
    a1*b2+b1*a2+c1*d2-d1*c2;
    a1*c2-b1*d2+c1*a2+d1*b2;
    a1*d2+b1*c2-c1*b2+d1*a2];
end

