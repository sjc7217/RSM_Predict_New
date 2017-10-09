from netCDF4 import Dataset
import multiprocessing
import torch
from torch.autograd import Variable
import csv
import numpy

#
#用于整体网络数据误差ME，MB等误差计量值的计算
#

#初始化所有网络参数进入内存，减少IO读取加快速度
def init_():
    for x in range(174):
        for y in range(150):
            name = "/media/lordshi/Life/RSM_0921_ALL/net_" + str(x) + "_" + str(y) + ".pkl"
            net = torch.load(name)
            NET_LIST.append(net)

#单个验证情景网格拟合数据获取
def deal_one_situation(para_in):
    result = []
    for net in NET_LIST:
        predict = net(para_in)
        result.append(predict.data.numpy()[0][0])
    return result

#主调函数，根据table定义的外部验证数据获取网络你和数据
def get_30_situation():
    for i in range(30):
        parameter = table[i]

        #注意此处list的size，输入网络的数据需要二维
        para_float = [[float(i) for i in parameter]]
        # print(para_float)
        para_FT = torch.FloatTensor(para_float)
        para_in = Variable(para_FT)
        # print(para_in)
        #单调函数，获取每个情景数据
        one_situation = deal_one_situation(para_in)
        PREDICT_ALL.append(one_situation)

#用于CMAQ模拟数据获取，i定义了外部数据的index
def get_one_situation_rsm(i):
    res = []
    file_RSM_output = "./data/validate_input/ACONC.01.lay1.PM2.5." + str(403 + i)
    RSM_ = Dataset(file_RSM_output, "r", format="NETCDF4")
    for j in range(174):
        for k in range(150):
            res.append(RSM_.variables["PM25_TOT"][0, 0, j, k])
    return res

#计算ME，MB
def calculate_MB_and_ME():
    for i in range(30):
        predict = PREDICT_ALL[i]
        RSM_out = get_one_situation_rsm(i)
        bias_sum = 0
        error_sum = 0
        predict_sum = 0
        FB_sum = 0
        FE_sum = 0

        for index in range(len(predict)):
            bias_sum += (predict[index]-RSM_out[index])
            error_sum += abs(predict[index]-RSM_out[index])
            predict_sum += predict[index]
            FB_sum += (predict[index]-RSM_out[index])/(predict[index]+RSM_out[index])*2
            FE_sum += abs(predict[index] - RSM_out[index]) / (predict[index] + RSM_out[index]) * 2

        # mean_bias = sum([j - k for j in predict for k in RSM_out]) / 174 * 150
        # mean_error = sum([abs(j - k) for j in predict for k in RSM_out]) / 174 * 150
        mean_bias=bias_sum/(174*150)
        mean_error=error_sum/(174*150)
        mean_FB = FB_sum/(174*150)
        mean_FE = FE_sum/(174*150)

        #print(mean_bias)
        MB_RESULT.append(mean_bias)
        ME_RESULT.append(mean_error)
        NMB_RESULT.append(bias_sum/predict_sum)
        NME_RESULT.append(error_sum/predict_sum)

        MFB_RESULT.append(mean_FB)
        MFE_RESULT.append(mean_FE)

def mean(a):
    return sum(a)/len(a)


if (__name__ == "__main__"):
    #全局数据，保存网络
    NET_LIST=[]

    #验证情景获取
    file_factor = "./data/400validate.csv"
    file = open(file_factor, 'r')
    reader = csv.reader(file)
    table = [row[1:] for row in reader][1:]
    # print(table)

    #网络拟合值
    PREDICT_ALL = []

    #Mean Bias值集合
    MB_RESULT = []

    #Mean Error值集合
    ME_RESULT = []

    NMB_RESULT = []

    NME_RESULT = []

    MFB_RESULT = []

    MFE_RESULT = []

    #主要调用
    init_()
    get_30_situation()
    calculate_MB_and_ME()

    #print(PREDICT_ALL)
    print("MB_RESULT:", MB_RESULT)
    print("MB_RESULT_MEAN:", mean(MB_RESULT))
    print("MB_RESULT_MIN:", min(MB_RESULT))
    print("MB_RESULT_MAX:", max(MB_RESULT))

    print("------------------------------------------------------------------")

    print("ME_RESULT:", ME_RESULT)
    print("ME_RESULT_MEAN:", mean(ME_RESULT))
    print("ME_RESULT_MIN:", min(ME_RESULT))
    print("ME_RESULT_MAX:", max(ME_RESULT))

    print("------------------------------------------------------------------")

    print("NMB_RESULT:", NMB_RESULT)
    print("NMB_RESULT_MEAN:", mean(NMB_RESULT))
    print("NMB_RESULT_MIN:", min(NMB_RESULT))
    print("NMB_RESULT_MAX:", max(NMB_RESULT))

    print("------------------------------------------------------------------")

    print("NME_RESULT:", NME_RESULT)
    print("NME_RESULT_MEAN:", mean(NME_RESULT))
    print("NME_RESULT_MIN:", min(NME_RESULT))
    print("NME_RESULT_MAX:", max(NME_RESULT))

    print("------------------------------------------------------------------")

    print("MFB_RESULT:", MFB_RESULT)
    print("MFB_RESULT_MEAN:", mean(MFB_RESULT))
    print("MFB_RESULT_MIN:", min(MFB_RESULT))
    print("MFB_RESULT_MAX:", max(MFB_RESULT))

    print("------------------------------------------------------------------")

    print("MFE_RESULT:", MFE_RESULT)
    print("MFE_RESULT_MEAN:", mean(MFE_RESULT))
    print("MFE_RESULT_MIN:", min(MFE_RESULT))
    print("MFE_RESULT_MAX:", max(MFE_RESULT))