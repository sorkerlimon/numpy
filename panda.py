import numpy as np
import pandas as pd

data = np.arange(5000).reshape(100,50)
print(data)
columName = ["Column_"+str(number) for number in range(50)]
print(columName)

sampleData = pd.DataFrame(data)
print(sampleData)


