import numpy as np
import pandas as pd


#data_output=open('bitcoin_10e4.txt','a')

#data=[]
#i=0

#with open('../txedges-master/txedges.dat') as f:
#    d=f.readlines()
#    print (d)

#print (mat_varables[0])

#data=np.loadtxt('../txedges-master/txedges.dat')


chunksize=10**4

chunk_idx=0
list=[]




for chunk in pd.read_csv('../txedges-master/txedges.dat',chunksize=chunksize):
    data=chunk.to_csv(header=None)
    print (data)
    rows=data.split("\n")
    idx = 0
    for row in rows:
        need=np.fromstring(row,dtype=int,sep=',')
        if idx<chunksize:
            #data_output.write((str(need[1]) + ',' + str(need[2]) + '\n'))

            if need[1] not in list:
                list.append(need[1])
            if need[2] not in list:
                list.append(need[2])
        idx = idx+1
    chunk_idx = chunk_idx +1
    #print (chunk_idx)
    list_array=np.asarray(list)
    #print (list_array.shape)
