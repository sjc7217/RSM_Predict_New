import torch
from torch.autograd import Variable
import torch.nn.functional as F
import csv
import numpy as np

f = open("./data/output/out_82_94.csv","r")
reader = csv.reader(f)
para=[]
response=[]
for line in reader:
    para.append([float(i) for i in line[1:-1]])
    response.append(float(line[-1]))

factor =torch.FloatTensor(para)
res = torch.FloatTensor(response)

x,y= Variable(factor),Variable(res)

net = torch.nn.Sequential(
    torch.nn.Linear(30,20),
    torch.nn.Sigmoid(),
    torch.nn.Linear(20,10),
    torch.nn.Sigmoid(),
    torch.nn.Linear(10,1)
)


optimizer = torch.optim.Adagrad(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

for t in range(10000):
    prediction = net(x)     # input x and predict based on x

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
    print(loss.data.numpy()[0])
#print(prediction.data.numpy())
#net2 = torch.load("./data/net_saved/net_82_94.pkl")
#print(net2(x).data.numpy())
#torch.save(net,"./data/net_saved/net_82_94.pkl")

    #print(prediction.data.numpy())
    #print(np.shape(prediction.data.numpy()))
    #print(y.data.numpy())
