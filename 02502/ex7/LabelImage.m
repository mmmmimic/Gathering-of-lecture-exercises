function Ilabel = LabelImage( I, T1, T2, T3, T4, T5 )
%LabelImage Turn ranges into labels

Ilabel = I;

[rmax, cmax] = size(I);

for r=1:rmax;
    for c=1:cmax;
        if I(r,c) < T1,
            Ilabel(r,c) = 0;
        end
        if I(r,c) >= T1 && I(r,c) < T2,
            Ilabel(r,c) = 1;
        end
        if I(r,c) >= T2 && I(r,c) < T3,
            Ilabel(r,c) = 2;
        end
        if I(r,c) >= T3 && I(r,c) < T4,
            Ilabel(r,c) = 3;
        end
        if I(r,c) >= T4 && I(r,c) < T5,
            Ilabel(r,c) = 4;
        end
        if I(r,c) >= T5,
            Ilabel(r,c) = 5;
        end
    end
end


