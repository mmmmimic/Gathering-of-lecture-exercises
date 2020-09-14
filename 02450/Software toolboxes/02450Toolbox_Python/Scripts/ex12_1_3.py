# ex12_1_3
import numpy as np
# Load data. There is probably a library-way to parse the file but we will take the scenic route
with open('../Data/courses.txt','r') as f:
    D = f.read()
    print("Raw data matrix is:")
    print(D)
D = [ [int(x) for x in ds.split(",")] for ds in D.split('\n') if len(ds) > 0]
N = len(D)
M = max( [max(v) for v in D])
X = np.zeros( (N,M))
for i in range(N):
    d_m1 = [j-1 for j in D[i]]
    X[i,d_m1] = 1

# We should now have the correct binary data matrix:
labels = ["02322", "02450", "02451", "02453", "02454", "02457", "02459", "02582"]
print("Transformed data matrix X is:")
print(labels)
print(X)
print("All-done!")