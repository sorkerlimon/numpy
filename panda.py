import numpy as np
import pandas as pd

data = np.arange(5000).reshape(100,50)
# print(data)
columName = ["Column_"+str(number) for number in range(50)]
# print(columName)
#
# sampleData = pd.DataFrame(data) ### numpy data added into panda data frame
# print(sampleData)
#
# head = sampleData.head()     ##### First 5 line show
# print(head)
#
# datatype = type(sampleData)  ### Data type
# print(datatype)
#
# firstIndex = sampleData[0]    ## Index acces
# print(firstIndex)

columNameadded = pd.DataFrame(data,columns=columName)
print(columNameadded)

