# load number of transactions in  each block, we can use this to separate transactions by block.

import numpy as np
import pandas as pd
import csv

chunksize=351084

data_output=open('tx_perblock.txt','a')

for chunk1, chunk2 in zip(pd.read_csv('../txedges-master/txedges.dat', chunksize=chunksize),pd.read_csv('../bh.dat', chunksize=chunksize)):

    #print (chunk1)
    #print (chunk2)
    #data1 = chunk1.to_csv(header=None)
    #rows1 = data1.split("\n")


    data2 = chunk2.to_csv(header=None)
    rows2 = data2.split("\n")

    print (rows2)

    for row2 in rows2:
        #print (row2)
        element=row2.split("\t")
        #print (element[3])
        print (element)
        data_output.write(str((element[3])+'\n'))


        #need1 = np.fromstring(row2, dtype=int, sep=' ')
        #print (need1)

    #
    #
    #     need1 = np.fromstring(row1, dtype=int, sep=',')
    #     need2 = np.fromstring(row2, dtype=int, sep=',')
    #
    #     #print (need1)
    #
    #
    #     #print(need2[1])
    #     data_output.write((str(need2[1]) + ' ' + str(need1[1]) + ' ' + str(need1[2]) + '\n'))
    #     #break
    #     count = count+1
    #     print (count)
