# exercise 7.4.3
import numpy as np

# Load list of names from files
fmale = open('../Data/male.txt','r')
ffemale = open('../Data/female.txt','r')
mnames = fmale.readlines(); fnames = ffemale.readlines();
names = mnames + fnames
gender = [0]*len(mnames) + [1]*len(fnames)
fmale.close(); ffemale.close();

# Extract X, y and the rest of variables. Include only names of >4 characters.
X = np.zeros((len(names),4))
y = np.zeros((len(names),1))
n=0
for i in range(0,len(names)):
    name = names[i].strip().lower()
    if len(name)>3:
        X[n,:] = [ord(name[0])-ord('a')+1, ord(name[1])-ord('a')+1, ord(name[-2])-ord('a')+1, ord(name[-1])-ord('a')+1]
        y[n,0] = gender[i]
        n+=1
X = X[0:n,:]; y = y[0:n,:];

N, M = X.shape; C = 2
attributeNames = ['1st letter', '2nd letter', 'Next-to-last letter', 'Last letter']
classNames = ['Female', 'Male'];

print('Ran Exercise 7.2.3')