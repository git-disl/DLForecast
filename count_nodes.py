import numpy as np
import pandas as pd
#data=open('./evalne/tests/data/bitcoin_10e4.txt','r')


output=[]
idx=0
row_idx=0

in_addr=[]
out_addr=[]
trans=[]

i=0
shared=0
chunksize=10000

#for row in data:

for chunk in pd.read_csv('./evalne/tests/data/bitcoin_full.txt', chunksize=chunksize, skiprows=2536161805):
    data = chunk.to_csv(header=None)
    rows = data.split("\n")
    #print (data)
    for row in rows:
        eed = np.fromstring(row, dtype=int, sep=',')
        try:
            if  eed[1] not in output:
                output.append(eed[1])
                idx=idx+1
            if eed[1] not in in_addr:
                in_addr.append(eed[1])

            if eed[2] not in output:
                output.append(eed[2])
                idx = idx + 1
            if eed[2] not in out_addr:
                out_addr.append(eed[2])
            row_idx = row_idx + 1
        except:
            continue
    i=i+1
    print (i)
    if i*chunksize+1>90000:
        break



for ele_in in in_addr:
    if ele_in in out_addr:
        shared = shared + 1


#print (idx)
#print (row_idx)




print ('both in and out')
print (shared)

print ('in only')
print (len(in_addr)-shared)

print ('out only')
print (-shared+len(out_addr))