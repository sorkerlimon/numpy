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

# dataCollect = np.arange(50).reshape(10,5)
# print(dataCollect)

# columname = ["Column_"+str(name) for name in  range(5)]
# print(columname)

# data_and_column = pd.DataFrame(dataCollect,columns=columname)
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


data = pd.read_csv("C:/Users/21100002/Desktop/Pandas/Indian Data Set/indian_liver_patient.csv")
# print(data)
# print(data.head(10))
# print(data.tail(10))
# print(data.columns)
# print(data.dtypes)
# print(data.describe()) ## All valu details like count, mean, std ,min,max
# print(data.index)  ### for check start valu end value

# print(data["Age"])
# print(type(data["Age"]))


# print(data["Direct_Bilirubin"])
# print(type(data["Direct_Bilirubin"]))     ### Acces one column


# print(data[["Total_Bilirubin","Direct_Bilirubin"]])   #### Access  2 colum like that
 # print(data[50:60])  ######## Work on index

############# Label ways acces

# print(data.loc[1])  ### It,s comes line series way
# print(data.loc[[1]])  ### It,s comes line tabuler way
# print(data.loc[:3,["Age","Gender"]])


########################## Filtering and selection data

# data = pd.read_csv("C:/Users/21100002/Desktop/Pandas/Indian Data Set/indian_liver_patient.csv")
# # print(data.head())
# #
# datafile = (data["Total_Bilirubin"].head(n=5)) > 2
# print(datafile)
#
#
# datafile2 = data["Total_Bilirubin"] < 2
# print(datafile[datafile2])



#
# #### Missing value
# data = pd.read_csv("C:/Users/21100002/Desktop/Pandas/Indian Data Set/indian_liver_patient.csv")
# # print(data.describe())
# print(data.isnull().sum())
# print("Total_Bilirubin",data["Total_Bilirubin"].isnull().sum())
#
# data["Total_Bilirubin"] = data["Total_Bilirubin"].fillna(0)
# print(data["Total_Bilirubin"])



##### Mathmatical Operations
#
# data = pd.read_csv("C:/Users/21100002/Desktop/Pandas/Indian Data Set/indian_liver_patient.csv")
# print(data["Total_Bilirubin"].sum())
# print(data["Total_Bilirubin"].var())
# print(data["Total_Bilirubin"].mean())



######## Map apply applymap in panda

data = pd.read_csv("C:/Users/21100002/Desktop/Pandas/Indian Data Set/indian_liver_patient.csv")
# seriesobj = data["Total_Bilirubin"]
# # print(seriesobj.head())
#
# pandasDFseriesobj = data[["Total_Bilirubin"]]
# # print(pandasDFseriesobj.head())
#
#
# print(seriesobj.map(str))
# print(seriesobj.map(str)[0])

#
# dictCheck = {"a":10,"b":20,"c":30}
# samNumpy = np.array(["a","b","c"])
# dataFSamp = pd.DataFrame(samNumpy,columns=["columName"])
# print(dataFSamp)
#
# dataFSamp1 = dataFSamp["columName"].map(dictCheck)
# print(dataFSamp1)
#
#
# def checkfunc(x):
#     return dictCheck[x]
#
# dataFSamp2 = dataFSamp["columName"].map(checkfunc)
# print(dataFSamp2)




### Apply with Lambda



