import multiprocessing
import csv
import os
import tensorflow as tf

'''
连续化处理整个空间网络的参数拟合

'''
class CMAQ_PREDICT:
    'CMAQ_PREDICT实例化类，运行run方法计算网格训练数据'

    # 构造函数，系数和训练参数初始化，实例化RSM_PREDICT类需要若干参数，分别为CUDA加速标志，计算区域定义（list），网络计算精度（出口），最少迭代次数出口，多进程并行计算核数，网络结构定义（list）
    def __init__(self,cuda_available,region_def,accuracy,times,process_num,net_structure):
        #CUDA加速标志
        self.CUDA_AVAILABLE = cuda_available
        #独立计算区域定义 list类型
        self.REGION_DEF = region_def
        #训练输入因子 368*30
        self.para = []
        # 神经网络MSE拟合程度度量值
        self.ACCURACY = accuracy
        #最低训练次数出口
        self.TIMES = times
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
        path = path+ "_12new_t"+str(self.TIMES)+"_a"+str(self.ACCURACY)
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


    # 训练代码
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


        try:
            input_ = tf.placeholder(dtype=tf.float32,shape=[None,30])
            tag_ = tf.placeholder(dtype=tf.float32,shape=[None,30])

            # l1_w = tf.Variable(tf.zeros([30, 30])+0.1)
            # l2_w = tf.Variable(tf.random_normal([30, 30]))
            # l3_w = tf.Variable(tf.random_normal([30, 30]))
            # l1_b = tf.Variable(tf.random_normal([30]))
            # l2_b = tf.Variable(tf.random_normal([30]))
            # l3_b = tf.Variable(tf.random_normal([30]))

            l1_w = tf.Variable(tf.zeros([30, 30]) + 0.1)
            l2_w = tf.Variable(tf.zeros([30, 30]) + 0.1)
            l3_w = tf.Variable(tf.zeros([30, 30]) + 0.1)
            l1_b = tf.Variable(tf.zeros([30])+0.1)
            l2_b = tf.Variable(tf.zeros([30])+0.1)
            l3_b = tf.Variable(tf.zeros([30])+0.1)

            # l1_w=tf.get_variable(name="l1_w",shape=[30,30],dtype=tf.float32)
            # l1_b=tf.get_variable(name="l1_b",shape=[30],initializer=tf.random_normal_initializer,dtype=tf.float32)
            #
            # l2_w=tf.get_variable(name="l2_w", shape=[30, 30], initializer=tf.truncated_normal_initializer,dtype=tf.float32)
            # l2_b=tf.get_variable(name="l2_b", shape=[30], initializer=tf.random_normal_initializer,dtype=tf.float32)

            res_l1 = tf.sigmoid(tf.matmul(input_,l1_w)+l1_b)
            res_l2 = tf.sigmoid(tf.matmul(res_l1, l2_w) + l2_b)
            res_l3 = tf.matmul(res_l2, l3_w) + l3_b

            loss = tf.losses.mean_squared_error(res_l3,tag_)
            optimizer = tf.train.AdagradOptimizer(0.4).minimize(loss)
            saver = tf.train.Saver()

        except Exception as e:
            print(e)


        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        for i in range(100):

            loss_res,_= sess.run([loss, optimizer], feed_dict={input_: self.para, tag_: response})
            print(loss_res)


        # saver.restore(sess, net_name)
        # loss_res = sess.run(loss, feed_dict={input_: self.para, tag_: response})
        # print(loss_res)
        saver.save(sess, net_name,global_step=i)
        sess.close()
        print("------------------------")

    #multiprocessing主调函数，用于不断调用进程池中的任务
    def run_one_time_project(self,x_y):
        try:
            if(self.REGION_DEF[0]==1 and self.REGION_DEF[1]==1):
                f = open("./data/output_12_new/out_" + str(x_y[0]) + "_" + str(x_y[1]) + ".csv", "r")
            else:
                f = open("./data/output_"+str(self.REGION_DEF[0])+"_"+str(self.REGION_DEF[1])+"_new/out_" + str(x_y[0]) + "_" + str(x_y[1]) + ".csv", "r")
            name = "./tf_net/net_" + str(x_y[0]) + "_" + str(x_y[1]) + ".ckpt"
            # 训练开启代码
            self.train(name, f)
            f.close()
        except:
            print("Can't open file or reach the bottom!")

    #运行入口
    def run(self):
        pool = multiprocessing.Pool(self.PROCESS_NUM)
        # 进程池中的进程依次调用可迭代对象进行计算
        pool.map(self.run_one_time_project, self.LIST_NUM[450:451])
        # 进程池不再添加新的进程
        pool.close()
        # 主线程阻塞等待子线程结束
        pool.join()

        print("Finished all jobs and quit the program!")

if(__name__=="__main__"):
    #实例化RSM_PREDICT类需要若干参数，分别为CUDA加速标志，计算区域定义（list），网络计算精度（出口），多进程并行计算核数，网络结构定义（list）
    predict_object = CMAQ_PREDICT(True,[6,5],0.5,1000,1,[30,50,40,30])
    predict_object.run()
