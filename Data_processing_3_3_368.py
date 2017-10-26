from netCDF4 import Dataset
import numpy as np
import csv

'''
用于将CMAQ输出的netcdf文件中的数据提取到格式化的csv格式中
所获得的csv文件 共有368行，分别是每个区域（共58*50个区域）
中训练的数据 

每行中前30个是当前污染物削减系数 后9个为3*3区间的数据值
'''

#所有netCDF文件的路径列表
PATHS=["./data/input_new_12/ACONC.01.lay1.PM2.5."+str(i) for i in range(1,369)]

CMAQ_RESULT_LIST=[]

#368个减排因子获取
try:
    fin=open("./data/control_0904.csv","r")
    reader=csv.reader(fin)
except:
    print("Can't find the correct file!")
    exit(1)
table=[row for row in reader]



def init():
    #获取全部netCDF文件的句柄，减少IO操作
    for path in PATHS:
        try:
            CMAQ_RESULT_LIST.append(Dataset(path, "r", format="NETCDF4"))
        except:
            print("Can't find CMAQ result file!")
            exit(1)



def get_i_j_file():
    #以3*3为小区域进行神经网络输出值进行拟合，适当考虑模型的区域相关关系
    for i in range(58):
        for j in range(50):
            #小区域在原174*150区域中的x，y坐标值
            x_start = 3*i
            y_start = 3*j

            case_item = []  #获取368*9训练数据
            for CMAQ_RESULT in CMAQ_RESULT_LIST:
                net_item= []  #获取9个小区域训练数据
                for m in range(x_start,x_start+3):
                    for n in range(y_start,y_start+3):
                        try:
                            num = CMAQ_RESULT.variables["PM25_TOT"][0, 0, m, n]
                        except:
                            print("There is not correct variable in netCDF file!")
                            exit(1)
                        net_item.append(num)
                case_item.append(net_item)

            try:
                fout=open("./data/output_3_3_new/out_"+str(i)+'_'+str(j)+".csv","w")
                writer=csv.writer(fout)
            except:
                print("Can't open file correctly!" )
                exit(1)

            #将减排因子和小区域模拟相结合生成输入文件
            for index in range(368):
                line=table[index]
                line_new = line+case_item[index]
                writer.writerow(line_new)
            fout.close()
    fin.close()

if(__name__=='__main__'):
    init()
    get_i_j_file()