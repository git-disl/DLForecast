import csv
import numpy as np
import pandas as pd



output=[]


'''
data_output=open('bitcoin_alpha.txt','a')

with open('../soc-sign-bitcoinalpha.csv',newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    for row in spamreader:
        out_row = np.fromstring(', '.join(row),dtype=int,sep=',')
        data_output.write(str(out_row[0])+','+str(out_row[1])+'\n')
'''




'''
# to process CSV files with hash address

head=0
output=[]
list=[]
idx=0


data_output=open('bitcoin_graph.txt','a')

with open('./evalne/tests/data/subgraph5251.csv',newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    for row in spamreader:
        if head>0:
            #print(', '.join(row))
            #output=np.hstack((output,', '.join(row)))
            output.append(', '.join(row))
        head = head + 1
    for i in range(len(output)):
        if  output[i][:34] not in list:
            list.append(output[i][:34])
            idx_tmp1 = idx
            idx=idx+1
        else:
            for j in range(len(list)):
                if list[j]==output[i][:34]:
                   idx_tmp1=j

        if output[i][-34:] not in list:
            list.append(output[i][-34:])
            idx_tmp2 = idx
            idx=idx+1
        else:
            for j in range(len(list)):
                if list[j] == output[i][-34:]:
                    idx_tmp2 = j
        data_output.write(str(idx_tmp1) + ',' + str(idx_tmp2)+'\n')


    print (len(list))
    print (len(output))
'''





load_data=open('bitcoin_swap_weighted_end.csv','r')
data_output=open('delta_data1_weight_end.txt','a')



raw_data=load_data.read()
tmp=raw_data.splitlines()

for i,row in enumerate(tmp):
    #row = np.fromstring(row,dtype=int,sep=' ')
    #if i <10000:

    if i<20000 and i>=10000:
        need = np.fromstring(row, dtype=int, sep=' ')
        #print(need)
        data_output.write(str(need[0]) + ',' + str(need[1]) + ',' + str(need[2]) + '\n')