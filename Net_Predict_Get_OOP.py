import torch
from torch.autograd import Variable
import csv
import os

'''
所生成网络外部预测值获取程序（已通用化）
'''

#预测值获取类
class Net_Predict_Get:
    #构造函数，所需输入参数为：1.net_structure:网格结构字符串（数字之间用下划线连接），2.region_def:单个计算区域定义，为一个list
    def __init__(self, net_structure,region_def,cal_range):
        self.NET_LIST=[]
        self.NET_STRUCTURE = net_structure
        self.REGION_DEF = region_def
        self.CAL_RANGE = cal_range
        #自动创建输出文件保存目录
        if(not os.path.exists('./data/Net_Predict_'+self.NET_STRUCTURE)):
            os.makedirs('./data/Net_Predict_'+self.NET_STRUCTURE)

        #一次性载入所有网络值，提高运算速度
        for x in range(174//region_def[0]):
            for y in range(150//region_def[1]):
                try:
                    name = "./data/net_saved_"+net_structure+"_12new/net_" + str(x) + "_" + str(y) + ".pkl"
                    net = torch.load(name)
                    self.NET_LIST.append(net)
                except:
                    pass

        #外部验证因子获取
        file_factor = "./data/400validate.csv"
        file = open(file_factor, 'r')
        reader = csv.reader(file)
        self.table = [row[1:] for row in reader][1:]

    # 主调函数，根据table定义的外部验证数据获取网络你和数据
    def get_single_situation(self,varidate_num):

        parameter = self.table[varidate_num]
        # 注意此处list的size，输入网络的数据需要二维
        para_float = [[float(i) for i in parameter]]
        para_FT = torch.FloatTensor(para_float)
        para_in = Variable(para_FT)
        # 单调函数，获取每个情景数据
        one_situation_res = self.deal_one_situation_after_transfer(para_in)
        return one_situation_res

    #不同网格参数转换程序
    def get_small_net_i_j_m_n(self,x_y):
        x = x_y[0]
        y = x_y[1]
        res = []
        res.append(x // self.REGION_DEF[0])
        res.append(y // self.REGION_DEF[1])
        res.append(x % self.REGION_DEF[0])
        res.append(y % self.REGION_DEF[1])
        return res


    # 验证情景网格拟合数据获取
    def deal_one_situation_after_transfer(self,para_in):
        result_bf = []
        res = []
        for net in self.NET_LIST:
            predict = net(para_in)
            result_bf.extend(predict.data.numpy()[0])  # predict此处为30的向量，result变为一维向量

        #将绘图所需网格信息与当前计算值进行匹配
        for i in range(174):
            for j in range(150):
                small_net_index = self.get_small_net_i_j_m_n([i, j])
                index_whole = small_net_index[0] * (26100//(174//self.REGION_DEF[0])) + small_net_index[1] * (self.REGION_DEF[0]*self.REGION_DEF[1]) + \
                small_net_index[2] * self.REGION_DEF[1] + small_net_index[3]
                try:
                    res.append(result_bf[index_whole])
                except:
                    print(index_whole)
        return res

    #写入结果CSV文件
    def write_res_file(self,data, index):
        f = open('./data/Net_Predict_'+self.NET_STRUCTURE+'/NN_' + str(403 + index) + ".CSV", 'w', newline='')
        writer = csv.writer(f)
        for row in data:
            writer.writerow([row])
        f.close()

    #网络拟合值
    def run(self):
        for i in self.CAL_RANGE:
            single_res = self.get_single_situation(i)
            self.write_res_file(single_res,i)

#单文件运行入口
if(__name__=="__main__"):
    predict_object = Net_Predict_Get("30_500_725",[29,25],[0])
    predict_object.run()
