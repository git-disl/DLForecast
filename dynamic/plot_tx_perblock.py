import numpy as np
import matplotlib.pyplot as plt

data=open('../bh.dat','r')


output=np.zeros(508241)
out_idx=np.zeros(508241)
idx=0
row_idx=0

count=0

stat=np.zeros(6)
tmp=0

for row in data:
    #print (idx)

    if idx<=508241:
        out_idx[idx]=idx

        element=row.split("\t")
        output[idx]=int(element[3].strip('\n'))
        #print (output.max())
        count=count + int(element[3].strip('\n'))
        #print (int(element[3].strip('\n')))

        #if count == 39015:
        #  print (idx)
        if idx>508235:
            tmp=tmp+int(element[3].strip('\n'))
            break
        #   print (idx)

        # if idx==500000:
        #     print (element[2])
        #
        # if int(element[3].strip('\n'))==1:
        #     stat[0]=stat[0]+1
        # elif int(element[3].strip('\n'))<=10:
        #     stat[1] = stat[1] + 1
        # elif int(element[3].strip('\n'))<=100:
        #     stat[2] = stat[2] + 1
        # elif int(element[3].strip('\n')) <= 1000:
        #     stat[3] = stat[3] + 1
        # elif int(element[3].strip('\n')) <= 10000:
        #     stat[4] = stat[4] + 1
        # else:
        #     stat[5] = stat[5] + 1

    idx = idx + 1

print (tmp)
#print (count)
#print (output.min())

#print (stat)

# plt.plot(out_idx,output)
# plt.xlabel('block')
# plt.ylabel('tx per block')
# #plt.savefig('first100k_time.jpg')
# plt.show()







