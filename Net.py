import torch
from torch.autograd import Variable
import torch.nn.functional as F
import csv
import numpy

para=[]

def para_init():
    para_init_0= open("./data/output/out_0_0.csv", "r")
    reader = csv.reader(para_init_0)
    for line in reader:
        para.append([float(i) for i in line[1:-1]])


def train(net_name,filein):
    response=[]
    reader = csv.reader(filein)
    for line in reader:
        response.append(float(line[-1]))
    factor =torch.FloatTensor(para)
    res = torch.FloatTensor(response)

    x,y= Variable(factor).cuda(),Variable(res).cuda()

    net = torch.nn.Sequential(
        torch.nn.Linear(30,20),
        torch.nn.Sigmoid(),
        torch.nn.Linear(20,10),
        torch.nn.Sigmoid(),
        torch.nn.Linear(10,1)
    ).cuda()

    optimizer = torch.optim.Adagrad(net.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss


    for t in range(1000000):
        prediction = net(x)     # input x and predict based on x
        loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()    # apply gradients
        loss_value=loss.cpu().data.numpy()[0]
        if(loss_value<0.05):
            break
        print(loss_value)
        #print(prediction.cpu().data.numpy())
    try:
        torch.save(net.cpu(),net_name)
    except:
        print("Can't save the net"+net_name)
        exit(1)


def run_this():
    para_init()
    for x in range(174):
        for y in range(150):
            file="./data/output/out_"+str(x)+"_"+str(y)+".csv"
            net_name="./data/net_saved/net_"+str(x)+"_"+str(y)+".pkl"
            try:
                f = open(file, "r")
            except:
                print("Can't open file"+file)
                exit(1)
            train(net_name,f)
            f.close()


if(__name__=="__main__"):
    run_this()