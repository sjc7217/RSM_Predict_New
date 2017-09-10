from netCDF4 import Dataset
import multiprocessing
import torch
from torch.autograd import Variable
import csv
import numpy

#
#用于计算单个net网络在外部验证时的误差估算
#
#



#初始化全部计算出来的网络进入内存，加快运算速度
def init_():
    for x in range(174):
        for y in range(150):
            name = "./data/net_saved_multi_processing/net_" + str(x) + "_" + str(y) + ".pkl"
            net = torch.load(name)
            NET_LIST.append(net)


#按照外部验证清单处理每个网络结果
#返回值@result size为 外部验证数×网络数
def deal_nets(para_in):
     result=[]
     for net in NET_LIST:
         net_res=[]
         for para_once in para_in:
             predict = net(para_once).data.numpy()[0][0]
             net_res.append(predict)
         result.append(net_res)
     return result


# def deal_one_situation(para_in):
#     result = []
#     for net in NET_LIST:
#         predict = net(para_in)
#         result.append(predict.data.numpy()[0][0])
#     return result



#用于处理外部验证数据的格式化以及整个程序流程
def get_30_situation():
    situation_list = []
    for i in range(30):
        parameter = table[i]
        para_float = [[float(i) for i in parameter]]
        para_float=torch.FloatTensor(para_float)
        para_float=Variable(para_float)
        situation_list.append(para_float)
        # print(para_float)
    #situation_FT = torch.FloatTensor(situation_list)
    #para_in = Variable(situation_FT)
    #print(situation_list)
    #print(situation_list)

    #关键调用
    res= deal_nets(situation_list)
    return res


#获取RSM模拟值的相同size
def get_one_situation_rsm():
    res = []
    #RSM list全部提取出来，减少IO提高速度
    RSM_LIST=[]
    for m in range(30):
        file_RSM_output = "./data/validate_input/ACONC.01.lay1.PM2.5." + str(403 + m)
        RSM_LIST.append(Dataset(file_RSM_output, "r", format="NETCDF4"))

    #对于每个网络（空间点）获取RSM预测数据
    for k in range(174):
        for j in range(150):
            res_one_sit=[]
            for RSM in RSM_LIST:
                res_one_sit.append(RSM.variables["PM25_TOT"][0, 0, k, j])
            res.append(res_one_sit)
    return res

# def calculate_MB_and_ME():
#     for i in range(30):
#         predict = PREDICT_ALL[i]
#         RSM_out = get_one_situation_rsm(i)
#         bias_sum=0
#         error_sum=0
#         for index in range(len(predict)):
#             bias_sum+=(predict[index]-RSM_out[index])
#             error_sum+=abs(predict[index]-RSM_out[index])
#         # mean_bias = sum([j - k for j in predict for k in RSM_out]) / 174 * 150
#         # mean_error = sum([abs(j - k) for j in predict for k in RSM_out]) / 174 * 150
#         mean_bias=bias_sum/(174*150)
#         mean_error=error_sum/(174*150)
#         #print(mean_bias)
#         MB_RESULT.append(mean_bias)
#         ME_RESULT.append(mean_error)

#30个外部验证误差计算函数
def diff(pre,rsm):
    err=0
    for i in range(30):
        err+=abs(pre[i]-rsm[i])
    return err

#主程序入口
if (__name__ == "__main__"):
    #网络提取
    NET_LIST=[]

    #外部验证数据提取
    file_factor = "./data/400validate.csv"
    file = open(file_factor, 'r')
    reader = csv.reader(file)
    table = [row[1:] for row in reader][1:]

    #初始化代码
    init_()

    #网络预测值获取
    net_result=get_30_situation()
    #RSM获取
    rsm_result=get_one_situation_rsm()

    #误差计算
    error=[]
    for index in range(len(net_result)):
        predict_one=net_result[index]
        rsm_one=rsm_result[index]
        error_one=diff(predict_one,rsm_one)
        error.append(error_one)

    #print(type(error))
    print(error)

