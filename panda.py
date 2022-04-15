import numpy as np
import pandas as pd

# data = np.arange(5000).reshape(100,50)
# # print(data)
# columName = ["Column_"+str(number) for number in range(50)]
# # print(columName)
# #
# # sampleData = pd.DataFrame(data) ### numpy data added into panda data frame
# # print(sampleData)
# #
# # head = sampleData.head()     ##### First 5 line show
# # print(head)
# #
# # datatype = type(sampleData)  ### Data type
# # print(datatype)
# #
# # firstIndex = sampleData[0]    ## Index acces
# # print(firstIndex)
#
# columNameadded = pd.DataFrame(data,columns=columName) ## Column name added
# # print(columNameadded)
#
#
#
# # column = {"Name":"Limon","University":"Wub","Batch":"43/A"}
# # Row = {"salary":"2000","due":"1000","Payment":"2000"}
# #
# # datafile = pd.DataFrame(data=(column,Row))
# # print(datafile)
#
# name = ["limon","naw","rafi","shaown"]
# collectdata = {"Name":name,"Mark":np.random.randint(1,10,len(name))}
# datafile = pd.DataFrame(data=(collectdata))
# print(datafile)
#
# dicr1 = {"Name":list(range(10)),"name2":list(range(10,20))}
# result = pd.DataFrame(data=dicr1)
# print(result)

dataCollect = np.arange(50).reshape(10,5)
# print(dataCollect)

columname = ["Column_"+str(name) for name in  range(5)]
# print(columname)

data_and_column = pd.DataFrame(dataCollect,columns=columname)
# print(data_and_column)


# totallShape = data_and_column.shape
# print(totallShape)
# print(data_and_column.dtypes)
# print(data_and_column[1:3])
# colum =data_and_column.columns
# print(list(colum))

# collectData = ["Limon","Nawshin","Sadia","Sorker","Rafi","Sani","Palok"]
# totallmark = np.random.randint(70,80,len(collectData))
#
# universityResult = {"Name":collectData,"Result":totallmark,"Grade":"A+"}
# result = pd.DataFrame(data=universityResult)
# print(result)



