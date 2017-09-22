from netCDF4 import Dataset


for i in range(300):

    file_RSM_output = "/home/lordshi/PycharmProjects/RSM_PREDICT_NEW/data/Input/ACONC.01.lay1.PM2.5." + str(i+1)

    RSM = Dataset(file_RSM_output, "r", format="NETCDF4")
    print(i)
    try:
        res=RSM.variables["PM25_TOT"][0, 0, 44, 44]
        print(res)
    except:
        pass