import multiprocessing
import torch
from torch.autograd import Variable
import csv
import os

'''
连续化,区域化处理整个空间网络的参数拟合   
将整个174*150网格数据切分成3*3小块
所用神经网络初步定为30*40*30*9

'''
#训练系数初始值，只需要初始化一次
para = []
#神经网络MSE拟合程度度量值
ACCURACY = 0.2
#进程数
PROCESS_NUM = 2
#用于存放训练project的序号
LIST_NUM = []

para_init_0 = open("./data/output_3_3_new/out_0_0.csv", "r")
reader = csv.reader(para_init_0)
for line in reader:
    para.append([float(i) for i in line[1:31]])

#系数和训练参数初始化
for x in range(58):
    for y in range(50):
        if ((not os.path.exists("./data/net_saved_30_30_9new/net_" + str(x) + "_" + str(y) + ".pkl"))):
            LIST_NUM.append([x, y])


#训练代码，采用CUDA加速
def train(net_name,filein):

    response=[]
    global para
    #标记值获取并格式化
    reader = csv.reader(filein)
    for line in reader:
        response.append([float(i) for i in line[31:40]])  #取出每一行的训练数据，此处为2*2=4个

    factor = torch.FloatTensor(para)
    #print(factor)
    res = torch.FloatTensor(response)

    #cuda()表示此处启用了pytorch的GPU加速
    x,y= Variable(factor).cuda(),Variable(res).cuda()

    #网格结构定义
    net = torch.nn.Sequential(
        torch.nn.Linear(30,30),
        torch.nn.Sigmoid(),
        # torch.nn.Linear(40,30),
        # torch.nn.Sigmoid(),
        torch.nn.Linear(30,9)
    ).cuda()

    #优化器Adagrad
    optimizer = torch.optim.Adagrad(net.parameters(),lr=0.4)
    #误差函数MSEloss
    loss_func = torch.nn.MSELoss().cuda()  # this is for regression mean squared loss
    count=0
    #训练过程，反复调整参数，直到误差loss小于一定值
    while(1):
        count+=1
        #print(x)
        prediction = net(x).cuda()     # input x and predict based on x
        loss = loss_func(prediction, y).cuda()   # must be (1. nn output, 2. target)
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()    # apply gradients
        loss_value = loss.cpu().data.numpy()[0]

        #print(loss_value,net_name)
        if(loss_value<ACCURACY):
            print(count)
            break
        #print(loss_value)
        #print(prediction.cpu().data.numpy())
        #print(prediction.cpu().data.numpy())

    #保存网格
    try:
        torch.save(net.cpu(),net_name)
    except:
        print("Can't save the net"+net_name)
        exit(1)

    print("Saved "+net_name+" successfully!")

#主调函数，用于不断调用进程池中的任务
def run_one_time_project(x_y):
    try:
        f = open("./data/output_3_3_new/out_" + str(x_y[0]) + "_" + str(x_y[1])  + ".csv", "r")
        name = "./data/net_saved_30_30_9new/net_" + str(x_y[0]) + "_" + str(x_y[1]) + ".pkl"
        #训练开启代码
        train(name, f)
    except:
        print("Can't open file or reach the bottom!")
    f.close()

#主进程
if(__name__ == "__main__"):

    #LOCK = multiprocessing.Lock()
    #para_init()
    #开启进程数量为PROCESS_NUM的进程池
    pool = multiprocessing.Pool(PROCESS_NUM)
    #进程池中的进程依次调用可迭代对象进行计算
    pool.map(run_one_time_project,LIST_NUM)
    #进程池不再添加新的进程
    pool.close()
    #主线程阻塞等待子线程结束
    pool.join()

    print("Finished all jobs and quit the program!")
