import torch
from torch.autograd import Variable
import torch.nn.functional as F
import csv
import numpy
import matplotlib.pyplot as plt
para=[]

def para_init():
    para_init_0= open("./data/output/out_0_0.csv", "r")
    reader = csv.reader(para_init_0)
    for line in reader:
        para.append([float(i) for i in line[1:-1]])


def train(net_name,filein):
    loss_list=[]

    response=[]
    reader = csv.reader(filein)
    for line in reader:
        response.append(float(line[-1]))
    factor =torch.FloatTensor(para)
    res = torch.FloatTensor(response)

    x,y= Variable(factor).cuda(),Variable(res).cuda()

    net = torch.nn.Sequential(
        torch.nn.Linear(30,10),
        torch.nn.Sigmoid(),
        torch.nn.Linear(10,10),
        torch.nn.Sigmoid(),
        torch.nn.Linear(10,1)
    ).cuda()

    optimizer = torch.optim.Adagrad(net.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss(size_average=True)  # this is for regression mean squared loss


    while(1):
        prediction = net(x)     # input x and predict based on x
        loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()    # apply gradients
        loss_value=loss.cpu().data.numpy()[0]
        loss_list.append(loss_value)

        if(loss_value<0.05):
            break
        print(loss_value)
        #print(prediction.cpu().data.numpy())
    # try:
    #     torch.save(net.cpu(),net_name)
    # except:
    #     print("Can't save the net"+net_name)
    #     exit(1)
    return loss_list


def run_this():
    para_init()
    res=[]
    for x in range(56,57):
        for y in range(69,70):
            file="./data/output/out_"+str(x)+"_"+str(y)+".csv"
            net_name="./data/net_saved/net_"+str(x)+"_"+str(y)+".pkl"
            try:
                f = open(file, "r")
            except:
                print("Can't open file"+file)
                exit(1)
            res.append(train(net_name,f))
            f.close()
    return res

if(__name__=="__main__"):
    result=run_this()
    # print(result[0])
    # plt.figure()
    #
    # plt.plot(result[0])
    # plt.show()


