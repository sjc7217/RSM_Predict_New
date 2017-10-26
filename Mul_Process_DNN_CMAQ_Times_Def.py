import multiprocessing
import torch
from torch.autograd import Variable
import csv
import os

'''
连续化处理整个空间网络的参数拟合

'''
class CMAQ_PREDICT:
    'CMAQ_PREDICT实例化类，运行run方法计算网络训练数据'

    # 构造函数，系数和训练参数初始化，实例化RSM_PREDICT类需要若干参数，分别为CUDA加速标志，计算区域定义（list），计算迭代次数，多进程并行计算核数，网络结构定义（list）
    def __init__(self,cuda_available,region_def,run_times,process_num,net_structure):
        #CUDA加速标志
        self.CUDA_AVAILABLE = cuda_available
        #独立计算区域定义 list类型
        self.REGION_DEF = region_def
        #训练输入因子 368*30
        self.para = []
        # # 神经网络MSE拟合程度度量值
        # self.ACCURACY = accuracy

        self.RUN_TIMES = run_times
        # 进程数
        self.PROCESS_NUM = process_num
        # 用于存放训练project的序号
        self.LIST_NUM = []
        #网络结构定义
        self.NET_STRUCTURE = net_structure

        #自动生成网络结果存储目录
        path = "./data/net_saved"
        for net_cores in net_structure:
            path = path +"_" +str(net_cores)
        path = path + "new_" +str(self.RUN_TIMES)
        self.SAVED_PATH = path

        #创建存储文件夹
        if(not os.path.exists(self.SAVED_PATH)):
            os.makedirs(self.SAVED_PATH)

        #读入所有所需训练因子
        para_init_0 = open("./data/output_12_new/out_0_0.csv", "r")
        reader = csv.reader(para_init_0)
        for line in reader:
            self.para.append([float(i) for i in line[1:-1]])

        #排除已经存在的文件，建立运行所需数据pool
        for x in range(174//region_def[0]):
            for y in range(150//region_def[1]):
                if ((not os.path.exists(self.SAVED_PATH+"/net_" + str(x) + "_" + str(y) + ".pkl"))):
                    self.LIST_NUM.append([x, y])


    # 训练代码，采用CUDA加速
    def train(self, net_name, filein):
        response = []

        # 标记值获取并格式化
        reader = csv.reader(filein)
        for line in reader:
            if (self.REGION_DEF[0] == 1):
                response.append(float(line[-1]))
            else:
                response.append(
                    [float(i) for i in line[31:(31 + self.REGION_DEF[0] * self.REGION_DEF[1])]])  # 取出每一行的训练数据
        factor = torch.FloatTensor(self.para)
        res = torch.FloatTensor(response)

        # cuda()表示此处启用了pytorch的GPU加速
        if(self.CUDA_AVAILABLE):
            x, y = Variable(factor).cuda(), Variable(res).cuda()
        else:
            x, y = Variable(factor), Variable(res)

        # 网格结构定义
        net = self.net_produce()
        # 优化器Adagrad
        optimizer = torch.optim.Adagrad(net.parameters(), lr=0.4)
        # 误差函数MSEloss
        if(self.CUDA_AVAILABLE):
            loss_func = torch.nn.MSELoss().cuda()  # this is for regression mean squared loss
        else:
            loss_func = torch.nn.MSELoss()

        run_time = 0
        # 训练过程，反复调整参数，直到误差loss小于一定值
        while (run_time<self.RUN_TIMES):
            run_time += 1
            if(self.CUDA_AVAILABLE):
                prediction = net(x).cuda()  # input x and predict based on x
                loss = loss_func(prediction, y).cuda()  # must be (1. nn output, 2. target)
            else:
                prediction = net(x)
                loss = loss_func(prediction, y)
            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            if(self.CUDA_AVAILABLE):
                loss_value = loss.cpu().data.numpy()[0]
            else:
                loss_value = loss.data.numpy()[0]
        # 保存网格
        try:
            if(self.CUDA_AVAILABLE):
                torch.save(net.cpu(), net_name)
            else:
                torch.save(net, net_name)
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
        if(self.CUDA_AVAILABLE):
            net = torch.nn.Sequential(*net_para).cuda()
        else:
            net = torch.nn.Sequential(*net_para)
        return net


    #multioricessing主调函数，用于不断调用进程池中的任务
    def run_one_time_project(self,x_y):
        try:
            if(self.REGION_DEF[0]==1 and self.REGION_DEF[1]==1):
                f = open("./data/output_12_new/out_" + str(x_y[0]) + "_" + str(x_y[1]) + ".csv", "r")
            else:
                f = open("./data/output_"+str(self.REGION_DEF[0])+"_"+str(self.REGION_DEF[1])+"_new/out_" + str(x_y[0]) + "_" + str(x_y[1]) + ".csv", "r")
            name = self.SAVED_PATH+"/net_" + str(x_y[0]) + "_" + str(x_y[1]) + ".pkl"
            # 训练开启代码
            self.train(name, f)
            f.close()
        except:
            print("Can't open file or reach the bottom!")


    #运行入口
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
    # 构造函数，系数和训练参数初始化，实例化RSM_PREDICT类需要若干参数，分别为CUDA加速标志，计算区域定义（list），计算迭代次数，多进程并行计算核数，网络结构定义（list）
    predict_object = CMAQ_PREDICT(False,[1,1],500,2,[30,30,9])
    predict_object.run()