from netCDF4 import Dataset
import multiprocessing
import torch
from torch.autograd import Variable
import csv
import numpy

#
#用于计算单个net网络在外部验证（n=30时）的误差估算
#
#


#按照外部验证清单处理每个网络结果
def deal_net(para_in):
     net_res=[]
     for para_once in para_in:
         predict = net(para_once).data.numpy()[0][0]
         net_res.append(predict)
     return net_res

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
    res= deal_net(situation_list)
    return res


#获取RSM模拟值的相同size
def get_one_situation_rsm(x,y):
    RSM_LIST=[]
    for m in range(30):
        file_RSM_output = "./data/validate_input/ACONC.01.lay1.PM2.5." + str(403 + m)
        RSM_LIST.append(Dataset(file_RSM_output, "r", format="NETCDF4"))
    print(x,y)

    res_one_sit=[]
    for RSM in RSM_LIST:
        res_one_sit.append(RSM.variables["PM25_TOT"][0, 0, x, y])
    return res_one_sit

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
    path=input("请输入所需计算误差网格所在文件夹：")

    x=input("请输入网格x坐标：")

    x_index=int(x)

    y=input("请输入网格y坐标：")

    y_index=int(y)

    file_x_y=path+"net_"+x+"_"+y+".pkl"
    print(file_x_y)
    try:
        net = torch.load(file_x_y)
    except:
        print("Wrong path of net name!")
        exit(1)

    #外部验证数据提取
    file_factor = "./data/400validate.csv"
    file = open(file_factor, 'r')
    reader = csv.reader(file)
    table = [row[1:] for row in reader][1:]


    #网络预测值获取
    net_result=get_30_situation()
    print("net_result:",net_result)


    #RSM获取
    rsm_result=get_one_situation_rsm(x_index,y_index)
    print("rsm_result:",rsm_result)

    #误差计算
    error=[]

    for i in range(30):
        error.append(net_result[i]-rsm_result[i])

    print(error)
