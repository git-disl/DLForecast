import numpy as np

load_data=open('../deepwalk-master/example_graphs/karate.adjlist','r')
data_output=open('adjlist_graph.txt','a')



raw_data=load_data.read()

#tmp = np.fromstring(raw_data,dtype=int,sep='\n')
tmp=raw_data.splitlines()



for row in tmp:
    row = np.fromstring(row,dtype=int,sep=' ')
    for i in range(1,(len(row))):
        data_output.write(str(row[0]) + ',' + str(row[i]) + '\n')




