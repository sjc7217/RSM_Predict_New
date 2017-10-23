import torch
from torch.autograd import Variable
import csv
from netCDF4 import Dataset

'''
此程序用于对生成的模型的准确性进行评估，生成外部/内部工况下的模型预测值，并按照ArcGis软件的网格顺序排列输出
method:
    get_small_net_i_j_m_n函数负责将单个网格顺序中的2个定位参数（row,col）准变为区域网格中的4个参数
    deal_one_situation_after_transfer 将预测数据进行拼接与转换

'''

#初始化所有网络参数进入内存，减少IO读取加快速度
def init_():
    for x in range(29):
        for y in range(30):
            try:
                name = "./data/net_saved_30_40_40_30_6_5_new_500/net_" + str(x) + "_" + str(y) + ".pkl"
                net = torch.load(name)
                NET_LIST.append(net)
            except:
                pass


'''
@:param x_y :单个网格顺序中的2个定位参数（row,col）
'''
def get_small_net_i_j_m_n(x_y):
    x=x_y[0]
    y=x_y[1]
    res=[]
    res.append(x//6)
    res.append(y//5)
    res.append(x%6)
    res.append(y%5)
    return res

#验证情景网格拟合数据获取
def deal_one_situation_after_transfer(para_in):
    result_bf = []
    res=[]
    for net in NET_LIST:
        predict = net(para_in)
        result_bf.extend(predict.data.numpy()[0])     #predict此处为30的向量，result变为一维向量
    for i in range(174):
        for j in range(150):
            small_net_index = get_small_net_i_j_m_n([i,j])
            index_whole = small_net_index[0]*900+small_net_index[1]*30+small_net_index[2]*5+small_net_index[3]  #求出单一网格排序序号在区域网格排列中的顺序，以便进行转换
            try:
                res.append(result_bf[index_whole])
            except:
                print(index_whole)
    return res


# #用于CMAQ模拟数据获取，i定义了外部数据的index
# def get_one_situation_rsm(i):
#     res = []
#     file_RSM_output = "./data/validate_input_new/ACONC.01.lay1.PM2.5." + str(403 + i)
#     RSM_ = Dataset(file_RSM_output, "r", format="NETCDF4")
#     # 以6*5为小区域进行神经网络输出值进行拟合，适当考虑模型的区域相关关系         ##以大区域为单位对26100个格点进行按序排列，用来计算误差
#     for i in range(29):
#         for j in range(30):
#             # 小区域在原174*150区域中的x，y坐标值
#             x_start = 6 * i
#             y_start = 5 * j
#
#             for m in range(x_start, x_start + 6):
#                 for n in range(y_start, y_start + 5):
#                     try:
#                         num = RSM_.variables["PM25_TOT"][0, 0, m, n]
#                     except:
#                         print("There is not correct variable in netCDF file!")
#                         exit(1)
#                     res.append(num)
#     return res

#主调函数，根据table定义的外部验证数据获取网络你和数据
def get_single_situation():
    i=VARIDATE_NUM
    parameter = table[i]

    #注意此处list的size，输入网络的数据需要二维
    para_float = [[float(i) for i in parameter]]
    para_FT = torch.FloatTensor(para_float)
    para_in = Variable(para_FT)
    #单调函数，获取每个情景数据
    one_situation_res = deal_one_situation_after_transfer(para_in)
    return one_situation_res

#将结果写入csv文件，待ArcGis使用
def write_res_file(data):
    f=open('./data/Whole_Net_Value/result_1023.csv','w',newline='')
    writer = csv.writer(f)
    writer.writerow(['ID','PM2.5'])
    for ind,row in enumerate(data):
        writer.writerow([ind,row])
    f.close()

if (__name__ == "__main__"):

    #验证外部因子序号
    VARIDATE_NUM=5
    NET_LIST=[]

    init_()
    #验证情景获取
    file_factor = "./data/400validate.csv"
    file = open(file_factor, 'r')
    reader = csv.reader(file)
    table = [row[1:] for row in reader][1:]
    #网络拟合值
    single_res = get_single_situation()
    write_res_file(single_res)

    #deal_one_situation_after_transfer()