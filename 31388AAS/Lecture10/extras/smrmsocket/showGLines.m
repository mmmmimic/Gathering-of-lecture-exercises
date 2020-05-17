function showGLines( Lines,property)

k=0;
for i=1:size(Lines,1)
    if abs(Lines(i,3)-Lines(i,4))<10
        continue;
    end
    X= 1+[Lines(i,1).*cos(Lines(i,2))+Lines(i,3).*sin(Lines(i,2)) Lines(i,1).*cos(Lines(i,2))+Lines(i,4).*sin(Lines(i,2))];
    Y= 1+[Lines(i,1).*sin(Lines(i,2))-Lines(i,3).*cos(Lines(i,2)) Lines(i,1).*sin(Lines(i,2))-Lines(i,4).*cos(Lines(i,2))];
    if nargin == 1
        line(X,Y);
    else
        line(X,Y,'color',property);
    end 
    k=k+1;
end

display(['No of lines shown=' sprintf('%d',k)]);