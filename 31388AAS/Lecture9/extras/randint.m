function out = randint(rows, cols, range)

out = rand(rows,cols);
out = floor(out*((range(2)+1)-range(1)) + range(1));