from netCDF4 import Dataset
import multiprocessing
import torch
from torch.autograd import Variable
import csv
import numpy



def deal_one_situation(para_in):
    result = []
    for x in range(174):
        for y in range(150):
            name = "./data/net_saved_multi_processing/net_" + str(x) + "_" + str(y) + ".pkl"
            net=torch.load(name)
            predict=net(para_in)
            result.append(predict.data.numpy()[0])
    return result

def get_30_situation():
    for i in range(30):
        parameter=table[i]
        para_float=[[float(i) for i in parameter]]
        #print(para_float)
        para_FT=torch.FloatTensor(para_float)
        para_in=Variable(para_FT)
        #print(para_in)
        one_situation = deal_one_situation(para_in)
        PREDICT_ALL.append(one_situation)

def get_one_situation_RSM(i):
    res=[]
    file_RSM_output = "./data/validate_input/ACONC.01.lay1.PM2.5." + str(403 + i)
    RSM_ = Dataset(file_RSM_output, "r", format="NETCDF4")
    for j in range(174):
        for k in range(150):
            res.append(RSM_.variables["PM25_TOT"][0, 0, j, k])
    return res


def calculate_MB_and_ME():
    for i in range(30):
        predict=PREDICT_ALL[i]
        RSM_out=get_one_situation_RSM(i)
        mean_bias=sum([j - k for j in predict for k in RSM_out]) / 174 * 150
        mean_error = sum([abs(j - k) for j in predict for k in RSM_out]) / 174 * 150
        MB_RESULT.append(mean_bias)
        ME_RESULT.append(mean_error)

if(__name__=="__main__"):

    file_factor = "./data/400validate.csv"
    file=open(file_factor,'r')
    reader = csv.reader(file)
    table = [row[1:] for row in reader][1:]
    #print(table)
    PREDICT_ALL = []
    MB_RESULT = []
    ME_RESULT = []
    get_30_situation()
    print("",PREDICT_ALL)
    print("MB_RESULT:",MB_RESULT)
    print("MB_RESULT_MIN:",min(MB_RESULT))
    print("MB_RESULT_MAX:",max(MB_RESULT))
    
    print("ME_RESULT:", ME_RESULT)
    print("ME_RESULT_MIN:", min(ME_RESULT))
    print("ME_RESULT_MAX:", max(ME_RESULT))
