'''
此脚本用于生成用于QGIS直接读入的CMAQ-RSM-NN对比csv文件，使用了pandas模块读取和生成csv文件
'''
import pandas as pd

def cmaq_path_return(num):
	return "./Whole_Net_Value/CMAQ_"+str(num)+".CSV"

def rsm_path_return(num):
	return "./Whole_Net_Value/RSM_"+str(num)+".CSV"

def nn_path_return(num):
	return "./Net_Predict__30_40_40_30new_250/NN_"+str(num)+"_smoothized"+".CSV"

res = None

for i in range(403,432)[::2]:
	cmaq_res = pd.read_csv(cmaq_path_return(i))
	cmaq_res = cmaq_res.Values
	cmaq_res.name = "CMAQ_"+str(i)
	res = pd.concat([res,cmaq_res],axis=1)

	rsm_res = pd.read_csv(rsm_path_return(i))
	rsm_res = rsm_res.Values
	rsm_res.name = "RSM_"+str(i)
	res = pd.concat([res,rsm_res],axis=1)

	nn_res = pd.read_csv(nn_path_return(i),squeeze=True,header =None)
	nn_res.name = "NN_"+str(i)
	res = pd.concat([res,nn_res],axis=1)

res.to_csv('./CMAQ_RSM_NN_RESULT/result_30_40_40_30new_250.csv')