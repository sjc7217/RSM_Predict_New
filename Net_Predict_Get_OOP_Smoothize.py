import torch
from torch.autograd import Variable
import csv
import os

'''
所生成网络外部预测值获取程序（已通用化）
'''

#预测值获取类
class Net_Predict_Get:
    #构造函数，所需输入参数为：1.所需要的输入文件夹，2.region_def:单个计算区域定义，为一个list,3.cal_range:外部验证工况选取，list 4.是否进行smooth标志
    def __init__(self, folder,region_def,cal_range,smoothize_flag):
        self.NET_LIST=[]
        self.SAVED_FOLDER = folder
        self.REGION_DEF = region_def
        self.CAL_RANGE = cal_range
        self.SMOOTHIZE_FLAG = smoothize_flag
        self.OUT_DIR = './data/Net_Predict_'+self.SAVED_FOLDER[9:]

        #自动创建输出文件保存目录
        if(not os.path.exists(self.OUT_DIR)):
            os.makedirs(self.OUT_DIR)

        #一次性载入所有网络值，提高运算速度
        for x in range(174//region_def[0]):
            for y in range(150//region_def[1]):
                try:
                    name = "./data/"+self.SAVED_FOLDER+"/net_" + str(x) + "_" + str(y) + ".pkl"
                    net = torch.load(name)
                    self.NET_LIST.append(net)
                except:
                    pass

        #外部验证因子获取
        file_factor = "./data/400validate.csv"
        file = open(file_factor, 'r')
        reader = csv.reader(file)
        self.table = [row[1:] for row in reader][1:]

    # 主调函数，根据table定义的外部验证数据获取网络数据
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

    #用于将计算结果进行平滑处理，特别是边界值   平滑处理方法：每个中间网格的浓度值由当前网格与周围网格值相应系数相加所得
    def smoothize_def(self, data_before_smoothize):
        res= data_before_smoothize.copy()
        for i in range(1, 173):
            for j in range(1, 149):
                res[i][j] = data_before_smoothize[i][j] * 0.9 + \
                            data_before_smoothize[i][j + 1] * 0.0125 + \
                            data_before_smoothize[i][j - 1] * 0.0125 + \
                            data_before_smoothize[i + 1][j] * 0.0125 + \
                            data_before_smoothize[i - 1][j] * 0.0125 + \
                            data_before_smoothize[i + 1][j + 1] * 0.0125 + \
                            data_before_smoothize[i + 1][j - 1] * 0.0125 + \
                            data_before_smoothize[i - 1][j + 1] * 0.0125 + \
                            data_before_smoothize[i - 1][j - 1] * 0.0125
        return res


    # 验证情景网格拟合数据获取
    def deal_one_situation_after_transfer(self,para_in):
        result_bf = []
        # res = []
        for net in self.NET_LIST:
            predict = net(para_in)
            result_bf.extend(predict.data.numpy()[0])  # predict此处为30的向量，result变为一维向量
        #全零返回值创建
        res = [[0 for i in range(150)] for j in range(174)]

        #将绘图所需网格信息与当前计算值进行匹配
        for i in range(174):
            for j in range(150):
                small_net_index = self.get_small_net_i_j_m_n([i, j])
                index_whole = small_net_index[0] * (26100//(174//self.REGION_DEF[0])) + small_net_index[1] * (self.REGION_DEF[0]*self.REGION_DEF[1]) + \
                small_net_index[2] * self.REGION_DEF[1] + small_net_index[3]
                try:
                    # res.append(result_bf[index_whole])
                    res[i][j]=result_bf[index_whole]
                except:
                    print(index_whole)

        if self.SMOOTHIZE_FLAG:
            res = self.smoothize_def(res)
        return res

    #网络拟合值
    def run(self):
        for i in self.CAL_RANGE:
            single_res = self.get_single_situation(i)
            self.write_res_file(single_res,i)


    #写入结果CSV文件
    def write_res_file(self,data, index):
        if self.SMOOTHIZE_FLAG:
            f = open(self.OUT_DIR+'/NN_' + str(403 + index) + "_smoothized" + ".CSV", 'w', newline='')
        else:
            f = open(self.OUT_DIR + '/NN_' + str(403 + index) + ".CSV", 'w', newline='')
        writer = csv.writer(f)
        for row in data:
            for col in row:
                writer.writerow([col])
        f.close()

#单文件运行入口
if(__name__ == "__main__"):
    #输入参数为：1.所需要的输入文件夹，2.region_def:单个计算区域定义，为一个list,3.cal_range:外部验证工况选取，list 4.是否进行smooth标志
    predict_object = Net_Predict_Get("net_saved_30_40_40_30new_250", [6, 5], [i for i in range(30) if i % 2 == 0],True)
    predict_object.run()
