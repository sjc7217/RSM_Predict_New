import multiprocessing
import torch
from torch.autograd import Variable
import csv
import numpy

para=[]

# PROJECT_LIST=[]

PROJECT_INDEX = 0

PROJECT_NUM=174*150

LIST_NUM=[]
#
# NET_NAME_LIST=[]

#
#
# def project_generator():
#     for x in range(174):
#         for y in range(150):
#             yield "./data/output/out_" + str(x) + "_" + str(y) + ".csv"
#
# def net_name_generator():
#     for x in range(174):
#         for y in range(150):
#             yield "./data/net_saved_multi_processing/net_" + str(x) + "_" + str(y) + ".pkl"
#
# project_list = project_generator()
#
# name_list = net_name_generator()


def para_init():
    para_init_0= open("./data/output/out_0_0.csv", "r")
    reader = csv.reader(para_init_0)
    for line in reader:
        para.append([float(i) for i in line[1:-1]])

    for x in range(174):
        for y in range(150):
            LIST_NUM.append([x,y])
            # PROJECT_LIST.append("./data/output/out_" + str(x) + "_" + str(y) + ".csv")
            # NET_NAME_LIST.append("./data/net_saved_multi_processing/net_" + str(x) + "_" + str(y) + ".pkl")
    # print(LIST_NUM)
    # exit(0)


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
        #print(loss_value)
        #print(prediction.cpu().data.numpy())
    try:
        torch.save(net.cpu(),net_name)
    except:
        print("Can't save the net"+net_name)
        exit(1)

    print("Saved "+net_name+"successfully!")


def run_one_time_project(lock,name):
    # global name_list
    while(1):
        #print("This is process "+name)
        lock.acquire()
        global PROJECT_INDEX
        PROJECT_INDEX += 1
        lock.release()
        try:
            f = open("./data/output/out_" + str(LIST_NUM[PROJECT_INDEX-1][0]) + "_" + str(LIST_NUM[PROJECT_INDEX-1][1]) + ".csv", "r")
            name= "./data/net_saved_multi_processing/net_" + str(LIST_NUM[PROJECT_INDEX-1][0]) + "_" + str(LIST_NUM[PROJECT_INDEX-1][1]) + ".pkl"
            train(name, f)
        except:
            print("Can't open file or reach the bottom!")
            break
        finally:
            f.close()




if(__name__=="__main__"):
    LOCK = multiprocessing.Lock()
    para_init()
    # 初始化一个线程对象，传入函数counter，及其参数1000
    th1 = multiprocessing.Process(target=run_one_time_project,args=(LOCK,"PRO1",))
    th2 = multiprocessing.Process(target=run_one_time_project,args=(LOCK,"PRO2",))
    th3 = multiprocessing.Process(target=run_one_time_project,args=(LOCK,"PRO3",))
    th4 = multiprocessing.Process(target=run_one_time_project,args=(LOCK,"PRO4",))
    # 启动线程
    th1.start()
    th2.start()
    th3.start()
    th4.start()
    # 主线程阻塞等待子线程结束
    th1.join()
    th2.join()
    th3.join()
    th4.join()

    print("Finish all jobs and quit the program!")