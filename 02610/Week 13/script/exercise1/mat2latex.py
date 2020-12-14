import sys
from scipy import io as scio

data = scio.loadmat('data.mat')['X']
length = scio.loadmat('len.mat')['len'][0]

head = "\\begin{table}[h]\n\\begin{tabular}{|l|l|l|l|l|l|}\n\\hline\nMethods & steepest descent & Newton & BFGS & coordinate search & nonlinear conjugate gradient \\\\ \\hline"
endl = "\\end{tabular}\n\\end{table}"

line = "$x_{}$ &{}&{}&{}&{}&{}\\\\ \\hline"

number = ['0'] + ['(1,4)']*5

d = []
cnter = 0
for i in range(5):
    d.append(data[:, cnter:cnter+length[i]])
    cnter += length[i]

LINE = []

for i in range(9):
    number = []
    number.append(str(i))
    for j in range(5):
        try:
            pair = (round(d[j][0, i], 2), round(d[j][1, i], 2))
            number.append(str(pair))
        except:
            number.append(' ')
    LINE.append(line.format(*number))

latex = head + '\n' + '\n'.join(LINE) + '\n' + endl
print(latex)

print(d[1])