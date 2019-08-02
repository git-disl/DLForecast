import numpy as np

data=open('data9.txt','r')


output=[]
idx=0
row_idx=0



for row in data:
    eed = np.fromstring(row, dtype=int, sep=',')
    row_idx =row_idx+1
    if  eed[0] not in output:
        output.append(eed[0])
        idx=idx+1
    if eed[1] not in output:
        output.append(eed[1])
        idx = idx + 1

print (idx)
print (row_idx)