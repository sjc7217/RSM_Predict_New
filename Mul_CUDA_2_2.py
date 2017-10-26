import multiprocessing
import torch
from torch.autograd import Variable
import csv
import os

'''
连续化,区域化处理整个空间网络的参数拟合   
将整个174*150网格数据切分成2*2小块
(2*2网格作为拟合单元,CUDA加速版本)

'''

class RSM_PREDICT_2_2_CUDA:
    # 构造函数，系数和训练参数初始化，实例化RSM_PREDICT类需要若干参数，分别为网络计算精度（出口），多进程并行计算核数，网络结构定义（list）
    def __init__(self,accuracy,process_num,net_structure):
        self.para = []
        # 神经网络MSE拟合程度度量值
        self.ACCURACY = accuracy
        # 进程数
        self.PROCESS_NUM = process_num
        # 用于存放训练project的序号
        self.LIST_NUM = []
        # 网络结构定义
        self.NET_STRUCTURE = net_structure

        # 自动生成网络结果存储目录
        path = "./data/net_saved"
        for net_cores in net_structure:
            path = path + "_"+str(net_cores)
        path += "new"
        self.SAVED_PATH = path

        if (not os.path.exists(self.SAVED_PATH)):
            os.makedirs(self.SAVED_PATH)

        #读入所有所需训练因子
        para_init_0 = open("./data/output_12_new/out_0_0.csv", "r")
        reader = csv.reader(para_init_0)
        for line in reader:
            self.para.append([float(i) for i in line[1:-1]])

        # 系数和训练参数初始化
        for x in range(87):
            for y in range(75):
                if ((not os.path.exists(self.SAVED_PATH+"/net_" + str(x) + "_" + str(y) + ".pkl"))):
                    self.LIST_NUM.append([x, y])

    # 训练代码，采用CUDA加速
    def train(self,net_name, filein):

        response = []
        reader = csv.reader(filein)
        for line in reader:
            response.append([float(i) for i in line[31:35]])  # 取出每一行的训练数据，此处为2*2=4个

        factor = torch.FloatTensor(self.para)
        # print(factor)
        res = torch.FloatTensor(response)

        # cuda()表示此处启用了pytorch的GPU加速
        x, y = Variable(factor).cuda(), Variable(res).cuda()

        # 网格结构定义
        net = self.net_produce()

        # 优化器Adagrad
        optimizer = torch.optim.Adagrad(net.parameters(), lr=0.4)
        # 误差函数MSEloss
        loss_func = torch.nn.MSELoss().cuda()  # this is for regression mean squared loss
        while (1):
            prediction = net(x).cuda()  # input x and predict based on x
            loss = loss_func(prediction, y).cuda()  # must be (1. nn output, 2. target)
            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            loss_value = loss.cpu().data.numpy()[0]
            if (loss_value < self.ACCURACY):
                break

        # 保存网格
        try:
            torch.save(net.cpu(), net_name)
        except:
            print("Can't save the net" + net_name)
            exit(1)

        print("Saved " + net_name + " successfully!")


    #网络生成程序，调用输入的NET_STRUCTURE参数使用*[]可变参数形成网络
    def net_produce(self):

        net_struc=self.NET_STRUCTURE
        net_para=[]
        for ind,num in enumerate(net_struc[:-2]):
            net_para.append(torch.nn.Linear(num,net_struc[ind+1]))
            net_para.append(torch.nn.Sigmoid())
        net_para.append(torch.nn.Linear(net_struc[-2],net_struc[-1]))
        net = torch.nn.Sequential(*net_para).cuda()
        return net

    def run_one_time_project(self,x_y):
        try:
            f = open("./data/output_2_2_new/out_" + str(x_y[0]) + "_" + str(x_y[1]) + ".csv", "r")
            name = self.SAVED_PATH+"/net_" + str(x_y[0]) + "_" + str(x_y[1]) + ".pkl"
            # 训练开启代码
            self.train(name, f)
        except:
            print("Can't open file or reach the bottom!")
        f.close()

    def run(self):
        pool = multiprocessing.Pool(self.PROCESS_NUM)
        # 进程池中的进程依次调用可迭代对象进行计算
        pool.map(self.run_one_time_project, self.LIST_NUM)
        # 进程池不再添加新的进程
        pool.close()
        # 主线程阻塞等待子线程结束
        pool.join()

        print("Finished all jobs and quit the program!")


if(__name__=="__main__"):
    predict_object = RSM_PREDICT_2_2_CUDA(0.5,2,[30,30,10,4])
    predict_object.run()


