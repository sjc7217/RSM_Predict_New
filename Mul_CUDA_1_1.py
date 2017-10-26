import multiprocessing
import torch
from torch.autograd import Variable
import csv
import os

'''
连续化处理整个空间网络的参数拟合(单个网格作为拟合单元,CUDA加速版本)

'''
class RSM_PREDICT_1_1_CUDA:
    'RSM实例化类，运行run方法获取网格训练数据'

    # 构造函数，系数和训练参数初始化，实例化RSM_PREDICT类需要若干参数，分别为网络计算精度（出口），多进程并行计算核数，网络结构定义（list）
    def __init__(self,accuracy,process_num,net_structure):
        self.para = []
        # 神经网络MSE拟合程度度量值
        self.ACCURACY = accuracy
        # 进程数
        self.PROCESS_NUM = process_num
        # 用于存放训练project的序号
        self.LIST_NUM = []
        #网络结构定义
        self.NET_STRUCTURE = net_structure

        #自动生成网络结果存储目录
        path = "./data/net_saved_"
        for net_cores in net_structure:
            path = path + str(net_cores) + "_"
        path += "12new"
        self.SAVED_PATH = path

        if(not os.path.exists(self.SAVED_PATH)):
            os.makedirs(self.SAVED_PATH)

        #读入所有所需训练因子
        para_init_0 = open("./data/output_12_new/out_0_0.csv", "r")
        reader = csv.reader(para_init_0)
        for line in reader:
            self.para.append([float(i) for i in line[1:-1]])

        #排除已经存在的文件，建立运行所需数据pool
        for x in range(174):
            for y in range(150):
                if ((not os.path.exists(self.SAVED_PATH+"/net_" + str(x) + "_" + str(y) + ".pkl"))):
                    self.LIST_NUM.append([x, y])


    # 训练代码，采用CUDA加速
    def train(self, net_name, filein):
        response = []

        # 标记值获取并格式化
        reader = csv.reader(filein)
        for line in reader:
            response.append(float(line[-1]))
        factor = torch.FloatTensor(self.para)
        res = torch.FloatTensor(response)

        # cuda()表示此处启用了pytorch的GPU加速
        x, y = Variable(factor).cuda(), Variable(res).cuda()

        # 网格结构定义
        net = self.net_produce()
        # 优化器Adagrad
        optimizer = torch.optim.Adagrad(net.parameters(), lr=0.4)
        # 误差函数MSEloss
        loss_func = torch.nn.MSELoss().cuda()  # this is for regression mean squared loss

        # 训练过程，反复调整参数，直到误差loss小于一定值
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


    # 主调函数，用于不断调用进程池中的任务
    def run_one_time_project(self,x_y):
        try:
            f = open("./data/output_12_new/out_" + str(x_y[0]) + "_" + str(x_y[1]) + ".csv", "r")
            name = self.SAVED_PATH+"/net_" + str(x_y[0]) + "_" + str(x_y[1]) + ".pkl"
            # 训练开启代码
            self.train(name, f)
            f.close()
        except:
            print("Can't open file or reach the bottom!")


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
    predict_object = RSM_PREDICT_1_1_CUDA(1,2,[30,40,1])
    predict_object.run()