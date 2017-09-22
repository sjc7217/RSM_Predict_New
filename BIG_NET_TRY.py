import torch
from torch.autograd import Variable
import csv
import os
from netCDF4 import Dataset

CMAQ_ALL = []

#训练系数初始值，只需要初始化一次
para=[]

ACCURACY=1

def init_():
    for i in range(1,369):
        CMAQ_ONE = get_one_situation_CMAQ(i)
        CMAQ_ALL.append(CMAQ_ONE)

    para_init_0 = open("./data/output/out_0_0.csv", "r")
    reader = csv.reader(para_init_0)
    for line in reader:
        para.append([float(i) for i in line[1:-1]])



def get_one_situation_CMAQ(i):
    res=[]
    file_RSM_output = "./data/validate_input/ACONC.01.lay1.PM2.5." + str(i)
    RSM_ = Dataset(file_RSM_output, "r", format="NETCDF4")
    for j in range(174):
        for k in range(150):
            res.append(RSM_.variables["PM25_TOT"][0, 0, j, k])
    return res




def train(net_name,filein):
    response=[]

    #标记值获取并格式化
    reader = csv.reader(filein)
    for line in reader:
        response.append(float(line[-1]))
    factor = torch.FloatTensor(para)
    res = torch.FloatTensor(response)

    #cuda()表示此处启用了pytorch的GPU加速
    x,y= Variable(factor).cuda(),Variable(res).cuda()

    #网格结构定义
    net = torch.nn.Sequential(
        torch.nn.Linear(30,100),
        torch.nn.Sigmoid(),
        torch.nn.Linear(100,5000),
        torch.nn.Sigmoid(),
        torch.nn.Linear(5000,26100)
    ).cuda()

    #优化器Adagrad
    optimizer = torch.optim.Adagrad(net.parameters(),lr=0.4)
    #误差函数MSEloss
    loss_func = torch.nn.MSELoss().cuda()  # this is for regression mean squared loss

    #训练过程，反复调整参数，直到误差loss小于一定值
    while(1):
        prediction = net(x).cuda()     # input x and predict based on x
        loss = loss_func(prediction, y).cuda()     # must be (1. nn output, 2. target)
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()    # apply gradients

        loss_value = loss.cpu().data.numpy()[0]

        #print(loss_value,net_name)
        if(loss_value<ACCURACY):
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


if(__name__=="__main__"):
    init_()
    CMAQ_RES = torch.FloatTensor(CMAQ_ALL)
    CMAQ_RES = Variable(CMAQ_RES)

    factor = torch.FloatTensor(para)

    input_ = Variable(factor)







