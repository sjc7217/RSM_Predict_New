from netCDF4 import Dataset
import numpy as np
import csv
paths=["./data/Input/ACONC.01.lay1.PM2.5."+str(i) for i in range(1,369)]
fin=open("./data/control_0904.csv","r")
# fout=open("./data/output/")
reader=csv.reader(fin)
table=[row for row in reader]

for i in range(0,174):
    for j in range(0,150):
        index=0
        fout=open("./data/output/out_"+str(i)+'_'+str(j)+".csv","w")
        writer=csv.writer(fout)
        for path in paths:
            res = Dataset(path, "r", format="NETCDF4")
            num=res.variables["PM25_TOT"][0, 0, i, j]
            line=table[index]
            line_new=line.copy()
            index+=1
            line_new.append(str(num))
            #print(type(num))

            #print(line_new)
            writer.writerow(line_new)
        fout.close()
fin.close()
#     print(rrr.variables["PM25_TOT"][0,0,173,149])
# rrr.close()

#import tensorflow as tf




