import numpy as np
import matplotlib.pyplot as plt

load_data=open('bitcoin_swap_weighted.csv','r')


raw_data=load_data.read()
tmp=raw_data.splitlines()

output=[]
out_idx=[]

record=np.zeros(10)

for i,row in enumerate(tmp):

    if i<100000:
        need = np.fromstring(row, dtype=np.float32, sep=' ')
        out_idx.append(i)
        output.append(need[2])
        print ()
        if need[2] == 0:
            record[0]=record[0]+1
        elif need[2] > 0 and need[2] < 10:
            record[1] = record[1] + 1
        elif need[2] >= 10 and need[2] < 100:
            record[2] = record[2] + 1
        elif need[2] >= 100 and need[2] < 500:
            record[3] = record[3] + 1
        elif need[2] >= 500 and need[2] < 1000:
            record[4] = record[4] + 1
        elif need[2] >= 1000 and need[2] < 2000:
            record[5] = record[5] + 1
        elif need[2] >= 2000 and need[2] < 5000:
            record[6] = record[6] + 1
        elif need[2] >= 5000 and need[2] < 10000:
            record[7] = record[7] + 1
        elif need[2] >= 10000 and need[2] < 20000:
            record[8] = record[8] + 1
        else:
            record[9] = record[9] + 1
    #if i>65000 and i<70000:
        #print (need[2])


plt.tick_params(labelsize=15)
plt.plot(out_idx,output)
plt.xlabel('sender-receiver pair',size=15)
plt.ylabel('#bitcoins', size=15)
plt.savefig('font_time.jpg',bbox_inches='tight')
plt.show()



#print (record)
# output=np.asarray(output)
#
# print (np.max(output),np.min(output),np.mean(output))