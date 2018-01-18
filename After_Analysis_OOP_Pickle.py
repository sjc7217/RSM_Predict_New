from netCDF4 import Dataset
import torch
from torch.autograd import Variable
import csv
import pickle

#
#用于整体网络数据误差ME，MB计算
#

class After_Analysis(object):
    def __init__(self,region_def,dst):
        self.NET_LIST = []

        self.REGION_DEF = region_def
        # 验证情景获取
        file_factor = "./data/400validate.csv"
        file = open(file_factor, 'r')
        reader = csv.reader(file)
        self.table = [row[1:] for row in reader][1:]

        # 网络拟合值
        self.PREDICT_ALL = []

        # Mean Bias值集合
        self.MB_RESULT = []

        # Mean Error值集合
        self.ME_RESULT = []

        self.NMB_RESULT = []

        self.NME_RESULT = []

        self.MFB_RESULT = []

        self.MFE_RESULT = []

        f = open(dst,'rb')          #读入整个参数网络！
        all_net = pickle.load(f)
        for net in all_net:
            self.NET_LIST.append(net[2])


    # 验证情景网格拟合数据获取
    def deal_one_situation(self,para_in):
        result = []
        for net in self.NET_LIST:
            predict = net(para_in)
            result.extend(predict.data.numpy()[0])  # predict此处为30的向量，result变为一维向量
        return result

    # 主调函数，根据table定义的外部验证数据获取网络你和数据
    def get_30_situation(self):
        for i in range(30):
            parameter = self.table[i]

            # 注意此处list的size，输入网络的数据需要二维
            para_float = [[float(i) for i in parameter]]
            # print(para_float)
            para_FT = torch.FloatTensor(para_float)
            para_in = Variable(para_FT)
            # print(para_in)
            # 单调函数，获取每个情景数据
            one_situation = self.deal_one_situation(para_in)
            self.PREDICT_ALL.append(one_situation)

    # 用于CMAQ模拟数据获取，i定义了外部数据的index
    def get_one_situation_rsm(self,i):
        res = []
        file_RSM_output = "./data/validate_input_new/ACONC.01.lay1.PM2.5." + str(403 + i)
        RSM_ = Dataset(file_RSM_output, "r", format="NETCDF4")
        # 以6*5为小区域进行神经网络输出值进行拟合，适当考虑模型的区域相关关系         ##以大区域为单位对26100个格点进行按序排列，用来计算误差
        for i in range(174//self.REGION_DEF[0]):
            for j in range(150//self.REGION_DEF[1]):
                # 小区域在原174*150区域中的x，y坐标值
                x_start = self.REGION_DEF[0] * i
                y_start = self.REGION_DEF[1] * j

                for m in range(x_start, x_start + self.REGION_DEF[0]):
                    for n in range(y_start, y_start + self.REGION_DEF[1]):
                        try:
                            num = RSM_.variables["PM25_TOT"][0, 0, m, n]
                            res.append(num)
                        except:
                            print("There is not correct variable in netCDF file!")
                            exit(1)
        return res

    # 计算ME，MB
    def calculate_MB_and_ME(self):
        for i in range(30):
            predict = self.PREDICT_ALL[i]
            RSM_out = self.get_one_situation_rsm(i)
            bias_sum = 0
            error_sum = 0
            predict_sum = 0
            FB_sum = 0
            FE_sum = 0

            for index in range(len(predict)):
                bias_sum += (predict[index] - RSM_out[index])
                error_sum += abs(predict[index] - RSM_out[index])
                predict_sum += predict[index]
                FB_sum += (predict[index] - RSM_out[index]) / (predict[index] + RSM_out[index]) * 2
                FE_sum += abs(predict[index] - RSM_out[index]) / (predict[index] + RSM_out[index]) * 2

            # mean_bias = sum([j - k for j in predict for k in RSM_out]) / 174 * 150
            # mean_error = sum([abs(j - k) for j in predict for k in RSM_out]) / 174 * 150
            mean_bias = bias_sum / (174 * 150)
            mean_error = error_sum / (174 * 150)
            mean_FB = FB_sum / (174 * 150)
            mean_FE = FE_sum / (174 * 150)

            # print(mean_bias)
            self.MB_RESULT.append(mean_bias)
            self.ME_RESULT.append(mean_error)
            self.NMB_RESULT.append(bias_sum / predict_sum)
            self.NME_RESULT.append(error_sum / predict_sum)

            self.MFB_RESULT.append(mean_FB)
            self.MFE_RESULT.append(mean_FE)

    def mean(self,a):
        return sum(a) / len(a)

    def print_data(self):
        # print(PREDICT_ALL)
        print("MB_RESULT_MEAN:", self.mean(self.MB_RESULT))
        print("MB_RESULT_MIN:", min(self.MB_RESULT))
        print("MB_RESULT_MAX:", max(self.MB_RESULT))

        print("------------------------------------------------------------------")

        # print("ME_RESULT:", ME_RESULT)
        print("ME_RESULT_MEAN:", self.mean(self.ME_RESULT))
        print("ME_RESULT_MIN:", min(self.ME_RESULT))
        print("ME_RESULT_MAX:", max(self.ME_RESULT))

        print("------------------------------------------------------------------")

        # print("NMB_RESULT:", NMB_RESULT)
        print("NMB_RESULT_MEAN:", self.mean(self.NMB_RESULT))
        print("NMB_RESULT_MIN:", min(self.NMB_RESULT))
        print("NMB_RESULT_MAX:", max(self.NMB_RESULT))

        print("------------------------------------------------------------------")

        # print("NME_RESULT:", NME_RESULT)
        print("NME_RESULT_MEAN:", self.mean(self.NME_RESULT))
        print("NME_RESULT_MIN:", min(self.NME_RESULT))
        print("NME_RESULT_MAX:", max(self.NME_RESULT))

        print("------------------------------------------------------------------")

        # print("MFB_RESULT:", MFB_RESULT)
        print("MFB_RESULT_MEAN:", self.mean(self.MFB_RESULT))
        print("MFB_RESULT_MIN:", min(self.MFB_RESULT))
        print("MFB_RESULT_MAX:", max(self.MFB_RESULT))

        print("------------------------------------------------------------------")

        # print("MFE_RESULT:", MFE_RESULT)
        print("MFE_RESULT_MEAN:", self.mean(self.MFE_RESULT))
        print("MFE_RESULT_MIN:", min(self.MFE_RESULT))
        print("MFE_RESULT_MAX:", max(self.MFE_RESULT))

    def run(self):

        # 主要调用
        self.get_30_situation()
        self.calculate_MB_and_ME()
        self.print_data()

if(__name__ == "__main__"):
    analysis = After_Analysis([6,5],"./data/All_Net_Picklized/1516263818_30_40_40_30_100.pkl")
    analysis.run()
