
#https://www.youtube.com/watch?v=ZB7BZMhfPgk&t=1536s
import numpy as np
from timeit import default_timer as timer

# a = np.array([1,2,3,4])
# print(a)
# print(a.shape)
# print(a.dtype)
# print(a.ndim)
# print(a.size)
# print(a.itemsize)
# print(a[:2])
#
# a[0] = 10
# print(a)
#
# b = a * np.array([4,2,1,3])
# print(b)


# l = [1,2,3]
# a = np.array([1,2,3])
#
# l = l + [4]
# print(l)
# a = a + np.array(4)
# print(a)

# 12.6 min

# Dot product

# l1 = [1,2,3]
# l2 = [4,5,6]
# a = np.array(l1)
# b = np.array(l2)
#
# dot = 0
# for i in range(len(l1)):
#     dot += l1[i] * l2[i]
# print(dot)
#
#
# dot = np.dot(a,b)
# print(dot)
#
#
#
#
# list1 = [5,6,7,8]
# list2 = [9,10,11,5]
#
# mul = 0
# for i in range(len(list1)):
#     mul += list1[i] * list2[i]
# print(mul)
#
# lis3 = np.array(list1)
# lis4 = np.array(list2)
# mul2 = np.dot(lis3,lis4)
# print(mul2)

#
# list1 = [5,6,7,8]
# list2 = [9,10,11,5]
#
# a = np.array(list1)
# b = np.array(list2)
#
# dot = a @ b
# print(dot)


# a = np.random.randn(1000)
# b = np.random.randn(1000)
#
# A = list(a)
# B = list(b)
#
# T = 1000
#
# def dot1():
#     dot = 0
#     for i in range(T):
#         dot += A[i] * B[i]
#     return dot
#
# def dot2():
#     return np.dot(a,b)
#
# start = timer()
# for t in range(T):
#     dot1()
# end = timer()
# t1 = end - start
#
# start = timer()
# for t in range(T):
#     dot2()
# end = timer()
# t2 = end - start
#
#
# print("List Calculatiom ",t1)
# print("Dot",t2)
# print("Ratio ",t1/t2)
#


# 17.56


# a = np.array([[1,2,3],[3,4,5]])
# # b  = np.array([[1,2],[3,4]])
# # # print(a)
# # # print(a.shape)
# # #
#
# print(a[0:,2])
# # b = a.T

# # print(b)
# # print(b.shape)
# # print(b[1:,0])
# print(np.linalg.inv(b))
# c = np.diag(b)
# print(np.diag(c))

#
# b  = np.array([[1,2],[3,4],[2,6]])
# print(b)
# print(b.shape)
#
#
# # bool_index  = b > 2
# # print(bool_index)
# # print(b[bool_index])
# print(b[b > 2])
#
# c = np.where(b>2 , b,0)
# print(c)

   ###### Shape Method Start

# a = np.array([2,4,5,6,7,8,6])
# b = [1,3,6]
# print(a[b])
# 29 minutes
#
# a = np.arange(1,9)
# print(a)
# print(a.shape)
# b = a.reshape((2,4))
# print(b)
# print(b.shape)
#
# c = a.reshape((4,2))
# print(c)
# print(c.shape)
#
# d = np.array([[1,2],[3,4],[5,6],[7,8]])
# print(d)
# print(d.shape)

# e = np.arange(1,7)
# print(e)
# print(e.shape)
# f = e[np.newaxis, :]
# # f = e[:, np.newaxis] #Another process
# print(f)
# print(f.shape)


      ######## oncnitation array #########


# a = np.array([[1,2],[3,4]])
# # # print(a)
# # # print(a.shape)
# b = np.array([[5,6]])
# # c = np.concatenate((a,b),axis=0)
# c = np.concatenate((a,b),axis=None)
# print(c)
# print(c.shape)


# a = np.array([1,2,3,4])
# b = np.array([5,6,7,8])
#
# c = np.hstack((a,b))
# c = np.vstack((a,b))
# print(c)


######## Broadcasting numpy array #########
#
#
# x = np.array([[1,2,3],[4,5,6],[1,2,3],[4,5,6]])
# a = np.array([1,0,1])
# y = x + a
# print(y)


#38.28 Minutes

######## function and axis #########

# a = np.array([[7,8,9,10,11,12,13],[17,18,19,20,21,22,23]])
# print(a)
# print(a.sum(axis=0)) #### it,s calculate between two rows (7 + 17)
# print(a.sum(axis=None)) ## it's calcultae all list together
# print(a.sum(axis=1))  ### It's calculate each list of sum
# print(np.sum(a,axis=1))  ### It's calculate each list of sum

######## data types #########
#
# a = np.array([1,2],dtype=np.int64)
# print(a)
# print(a.dtype)


######## copy array #########

# a = np.array([1,2,3])
# b = a
# b = a.copy()## it's Acutally copy no chnage a value
# b[0] = 42
# print(b)


######## generating data #########

# a = np.zeros([2,2],dtype=np.int32)
# print(a)
#
#
# a = np.ones((2,2))
# print(a)
#
# a = np.full((2, 2),4.0,dtype=np.int64)
# print(a)
# a = np.eye(3)
# print(a)
#
# a = np.linspace(0,20,4)
# print(a)

######## Random Number generator

# a = np.random.random((3,2))
# print(a)
#
# a = np.random.randn(1000)
# print(a.mean(),a.var())
#
#
# # a = np.random.randint(3,10,size=(3,3))
# a = np.random.randint(10,size=(3,3))
# print(a)
#
# a = np.random.choice(5,size=10)
# print(a)
#
#
# a = np.random.choice([-5,-3,-2],size=10)
# print(a)



######### Linear Algenbra
# a = np.array([[1,2],[3,4]])
# eigenvalu, eivector = np.linalg.eig(a)
#
# # print(eigenvalu)
# # print(eivector)
#
# b = eivector[:,0] * eigenvalu[0]
# print(b)
#
# c = a @ eivector[:,0]
# print(c)
#
# print(np.allclose(b,c))

# A = np.array([[1,1],[1.5,4.0]])
# b = np.array([2200,5050])
# X = np.linalg.inv(A).dot(b)
# print(X)
#
# x = np.linalg.solve(A,b)
# print(x)
#



# output = np.ones((5,5))
# # print(output)
#
# z = np.zeros((3,3))
#
# z[1,1]=9
# print(z)
#
# output[1:4,1:4] = z
# print(output)


#
# a = np.array([1,2,3])
# b = a.copy()
# b[0] = 100
# print(b)
# print(a)


# a = np.array([1,2,3,4])
# print(a + 2)
#
# b = np.cos(a)
# print(b)

############## Linear algebra

# a = np.ones([2,3],dtype=np.int64)
# print(a)
# b = np.full([3,2],2,dtype=np.int64)
# print(b)
#
# c = np.matmul(a,b)
# print(c)



# stat = np.array([[1,2,3],[4,5,6],[0,0,0]])
# print(np.min(stat,axis=1))

#
# before = np.array([[1,2,3,4],[5,6,7,8]])
# print(before)
# after = before.reshape([4,2])
# print(after)
#
# data =np.genfromtxt("C:/Users/21100002/Desktop/Pandas/numpydata.txt", delimiter=",")
# filedata = data.astype("int32")
# print(filedata[filedata > 4])


# a  = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25],[26,27,28,29,30],])
# print(a)
# print(a[2:4,0:2])
# print(a[[0,4,5],3:])



#
# a  = np.arange(25).reshape(5,5)
# # print(a)
#
#
# b  = np.array([1,2,3,4,5])
# c  = np.array([6,7,8,9,4])
# # print(b**c)
# print(np.sin(b))
# print(b.fill(-4.8))
#
# a = np.array([[0,1,2,3],[10,11,12,13]])
# print(a)
# a[1,3] = -1
# # print(a[1,3:4])
# print(a)

# a = np.arange(25).reshape(5,5)
# print(a)
# #
# # print(a[::2])
#
# # print(a[3:,2:])
#
# # print(a[:,2])
# print(a[2::2,::2])

# a = np.arange(25).reshape(5,5)
# print(a)
#
# # red = a[:,1::2]
# # print(red)
# #
# # yellow = a[4:,:]
# # print(yellow)
#
# blue = a[1::2,:4:2]
# print(blue)

# import matplotlib.pyplot as plt
#
# img  = plt.imread("C:/Users/21100002/Downloads/s.jpg")
#
# top = img[0:2, 1:3]
# left = img[1:3, 0:2]
#
# blurred = (top + left) / 5
#
# right = img[0:2, 1:3]
# bottom = img[0:2, 1:3]
#
# plt.imshow(img,cmap=plt.cm.hot)
# plt.figure()
# plt.imshow(blurred,cmap=plt.cm.hot)
# plt.show()



# 48 minute


### Fancy indexing
#
# a = np.arange(0,80,10)
# print(a)
# indices = [2,6,-3]
# y = a[indices]
# print(y)


# a = np.arange(25).reshape(5,5)
# print(a)
#
# b = a[2:,[0,2,4]]
# # print(b)
#
# c = a[[0,1,2,3],
#       [1,2,3,4]]
# print(c)
#
# mask = np.array([1,0,1,0,1],dtype=bool)
# print(a[mask,2])

# 1.13 hrs



# a = np.arange(25).reshape(5,5)
# # print(a)
#
# b = a[[0,2,3,3],[2,3,1,4]]
# # print(b)
#
# c = a % 3
# # print(c)
# d = a % 3 == 0
# print(d)
# e = a[ a % 3==0]
# print(e)

############  Multi Dimensional Array calculation method

# a = np.array([[1,2,3],[4,5,6]])
# b = a.sum()
# print(b)
# a = np.array([[1,2,3],[4,5,6]])
# b = a.sum(axis=0)
# print(b)
# a = np.array([[1,2,3],[4,5,6]])
# b = a.sum(axis=1)
# print(b)


# a = np.ones((5,6))
# print(a)
# print(a.shape)

# 1.39 mi


# a = np.array([[1,2,3,9],[4,5,6,2]])
# print(np.argmax(a))
# print(a.argmax())

# a = np.array([1,2,2,6,6])
# print(a==a.max())
# print(np.where(a==a.max()))


# a = np.arange(6)
# print(a)
# b =a.reshape(2,3)
# print(b)
# c = b.T
# print(c)
# print(c.shape)
#
#
# c = b.T.strides
# print(type(c))

#https://www.youtube.com/watch?v=yKcTNDVQa0c


varA = list(range(10))
varB = list(range(10,20))
varC = list(range(20,30))
#
# print(varA)
# print(varB)
# print(varC)

dataInAll = [varA,varB,varC]
# sum = [sum(i) for i in dataInAll]
# print(sum)


sampData = np.array(dataInAll)
# print(sampData)
#
# print("size :",sampData.size)
# print("dimension :",sampData.ndim)
# print("shape :",sampData.shape)
# print("dtype ",sampData.dtype)
# print("flags ",sampData.flags)
# print("strides ",sampData.strides)
# print("strides ",sampData.max())
# print("strides ",sampData.itemsize)

# sampData2 = np.array([[.4,.5,.6],[.4,.5,.6]])
# print("size :",sampData2.size)
# print("dimension :",sampData2.ndim)
# print("shape :",sampData2.shape)
# print("dtype ",sampData2.dtype)
# print("flags ",sampData2.flags)
# print("strides ",sampData2.strides)
# print("strides ",sampData2.max())
# print("strides ",sampData2.itemsize)



# transpose = sampData.T
# print(transpose)

# sliceing = sampData[:,:]
# # print(sliceing)
# sliceing = sampData[0:2,3:]
# print(sliceing)
# sliceing = sampData[2][-2]
# print(sliceing)
# sliceing = sampData[::2,::2]
# print(sliceing)
#
# sliceing = sampData[1][6]
# print(sliceing)


# sampData3 = sampData
# print(sampData3)
# sampData3[1][1]=13
# print(sampData3)
# print(sampData)

# print("-------------------------------------")
# sampData4 = sampData.copy()
# print(sampData4)
# sampData4[1][1]=13
# print(sampData4)
# print(sampData)
#
# varA = list(range(10))
# varB = list(range(10,20))
# varC = list(range(20,30))
#
# allData = np.array([varA,varB,varC])
# # print(allData)
# # print(allData.shape)
# # print(np.sum(allData,axis=1))
# # print(np.sum(allData))
#
# randomSamData = np.random.random(6000)
# # print(randomSamData.size)
# # print(randomSamData)
# reshapeMatrix = randomSamData.reshape(200,30)
# print(reshapeMatrix)
# print(reshapeMatrix)
# print(reshapeMatrix.shape)

## create numpy array with random numbers and get the numbers closest to .50
# exampletest = np.random.random(25)
# # print(exampletest)
# reshapeData = exampletest.reshape(5,5)
# # print(reshapeData)
# closeData = abs(reshapeData-.5)
# print(closeData)
# closeData1 = np.argmin(closeData,axis=0)
# print(closeData1)
# closeData1 = np.argmin(closeData,axis=1)
# print(closeData1)

#### Select the top 5 columns on order of highest variance

# sampleData = np.random.random(6000)
# sampleData2 = sampleData.reshape(100,60)
# # print(sampleData2)
# # print(sampleData2.shape)
# variance = np.var(sampleData2)
# # print(variance)
# variance = np.var(sampleData2,axis=0)
# # print(variance)
#
#
# incOrder = np.argsort(variance)
# print(incOrder)
# print(incOrder[-5:])
# print(sampleData2[:,incOrder[-5:]])

### Boradcasting

# sampData4 = np.random.random(28).reshape(7,4)
# sampData5 = sampData4+5
# print(sampData5)
# sampData5 = sampData4+np.array([1,2,3,4])
# print(sampData5)


#### Matrix array

# aMatrix = np.array([[1,2],[3,4]])
# bMatrix = np.array([[7,8],[5,6]])
#
# mul = aMatrix* bMatrix
# print(mul)
#
# mul = aMatrix.dot(bMatrix)
# print(mul)
#
# mul = aMatrix @ bMatrix
# print(mul)


######## Handyt Function
# a = np.ones([5,5],dtype=int)
# print(a.shape)

### Dense Layer calculation using numpy
#
# inputLayer = np.random.random(20)
# # print(inputLayer)
# # print(inputLayer.shape)
#
# hiddednLayer = np.random.randn(500,20)
# # print(hiddednLayer)
# # print(hiddednLayer.shape)
#
# biasLayer = np.random.random(500)
# # print(biasLayer)
# # print(biasLayer.shape)
#
# output = np.dot(hiddednLayer,inputLayer)+biasLayer
# print(output[:10])
#
# activation = np.maximum(0,output)
# print(activation[:10])


#### file input out put

# savefile = np.array([1,2,3,4])
# np.save("C:/Users/21100002/Desktop/git/file",savefile)
#
# loadfile = np.load("C:/Users/21100002/Desktop/git/file.npy")
# print(loadfile)
#
# textfile = np.array([7,8,9,56])
# np.savetxt("C:/Users/21100002/Desktop/git/file1.txt",textfile)
#
# loadtxt = np.loadtxt("C:/Users/21100002/Desktop/git/file1.txt")
# print(loadtxt)



####  https://www.youtube.com/watch?v=eClQWW_gbFk ### Numpy tutorials
#
# N = 60 * 60 * 24 * 365
# prices = 200 + np.arange(N)/200
# print(prices[:5])
#
# avg = np.mean(prices)
# print(avg)


# oneDArray = np.array([1,2,3,4,5])
# print(oneDArray.ndim)
# print(oneDArray.size)
# print(oneDArray.dtype)
# print(oneDArray.shape)
# print(type(oneDArray))
#
# twoDArray = np.array([[3,5,6,7,8],
#                       [1,2,3,4,5]
#                       ])
# print(twoDArray.ndim)
# print(twoDArray.size)
# print(twoDArray.dtype)
# print(twoDArray.shape)
# print(len(twoDArray))
# print(type(twoDArray))


# cat = np.full(shape=(3,4),fill_value="cat")
# print(cat)
# print(cat.dtype)
#
#
# randint = np.random.randint(low=2,high=9,size=(2,3))
# print(randint)

# zoo = np.array([[[1,2],[3,4],[5,6]],
#                 [[1,2],[3,4],[5,6]]])
# print(zoo[0,:,1])
# zoo[0,:,1] = 5
# print(zoo)

# dailywts = 185 - np.arange(5*7)/5
# print(dailywts)
# first =dailywts[5::7]
# print(first)
#
# second =dailywts[6::7]
# print(second)
#
# third = (first + second)/2
# print(third)

# seed = np.random.seed(5555)
# gold = np.random.randint(low=0,high=10,size=(7,7))
# print("Gold : \n",gold)
#
#
# locs = np.array([[0,4],
#                  [2,2],
#                  [2,3],
#                  [5,1],
#                  [6,3]])
#
# # print("Location :\n",locs)
#
# #
# # print(gold[0,4])
# # print(gold[2,2])
# # print(gold[[0,2],[4,2]])
#
# # print(locs[:,0])
# # print(locs[:,1])
#
# print(gold[locs[:,0],locs[:,1]])



######## Bilboard pblm


# linespace = np.linspace(start=0,stop=10,num=5)
# print(linespace)
#
# linespace = np.linspace(start=[0,5],stop=[10,15],num=5)
# print(linespace)
#
# linespace = np.linspace(start=[0,5],stop=[10,15],num=5,axis=1)
# print(linespace)

# linespace = np.abs(np.linspace(start=[17,32],stop=[28,36],num=3,axis=1,dtype=int)-30)[[0,1],[2,0]]
# print(linespace)

## Bradcasting
#
# a = np.array([3,11,4,5])
# b = np.array([4,5,6])
#
# sum = np.sum([a,b,],axis=1)
# print(sum)

### Reshape concept


##### BOlean Indexing

# foo = np.array([[1,2,2],
#                 [2,1,2],
#                 [2,4,2]])
#
# # print(foo)
# mask = (foo == 2)
# # print(mask)
# foo[mask] = 0
# # print(foo)
#
#
# row1_and_3 = np.array([True,False,True])
# col2_and_3 = np.array([False,False,False])
# h = foo[row1_and_3,col2_and_3]
# print(h)



# names = np.array(["Denis","Dee","Charle","Mac","Franl"])
# ages = np.array([43,44,43,42,74])
# gender = np.array(["male","female","male",":male","male"])
#
# print(names[ages>=44])
#
# print(names[(gender=="male")&(ages>42)])
# print(names[(gender=="male")|(ages<43)])

#
# bot = np.ones([5,5])
# print(bot)
#
#
# bot[[0,2],[1,2]]=np.nan
# print(bot)
# seeed = np.random.seed(123)
# random = np.random.randint(1,9,5,dtype=int)
# print(random)




# foo = np.array([[1,2],[3,4],[5,6],[7,8],[9,10]])
# print(foo)
# print(foo.shape[0])
# np.random.seed(1234)
# rand_row = np.random.randint(
#     low=0,
#     high=foo.shape[0],
#     size=3
# )
# print(rand_row)
#
# generator = np.random.default_rng(seed=123)
# print(generator.integers(1,7,4))
# print(generator.choice(10,3,replace=True))


###### Love Score problem
#
# generator = np.random.default_rng(1010)
# love_scores = np.round(generator.uniform(0,100,10),2)
# print(love_scores)
# print(love_scores[:,None])
# print(love_scores[None,:])
#
#
#
# np.set_printoptions(linewidth=999)
# love_scores2 = np.abs(love_scores[:,None]-love_scores[None,:])
# # print(love_scores2)

#### vindictive problem

# generator = np.random.default_rng(80085)
# scores = np.round(generator.uniform(30,100,15))
# print(scores)
#
# scores[(scores<60).nonzero()[0][:3]]=0
# print(scores)


###### Egg problem solve

# field = np.zeros([10,10])
# print(field)
#
# genarator = np.random.default_rng(1234)
# print(genarator)
# vals = np.round(genarator.normal(size=20),2)
# print(vals)
#
# locs = genarator.choice(field.size,len(vals),replace=False)
# print(locs)
#
# field.ravel()[locs]=vals
# print(field)





#############

# squee = np.array([
#     [5.0,2.0,9.0],
#     [1.0,0.0,2.0],
#     [1.0,7.0,8.0]
# ])
# # print(squee)
# # print(np.sum(squee,axis=1))
# #
# squee[0,0]=np.nan
# # print(squee)
# # print(np.sum(squee))
# print(np.sum(squee,where=np.isnan(squee)))


# foo = np.array(["a","b"])
# bar = np.array(["c","d"])
# baz = np.array([["e","f"]])
# bingo = np.array([["g","h","i"]])
# bongo = np.array([["j","k"],
#                   ["l","m"]])
#
# vstack = np.vstack((foo,bar,baz))
# print(vstack)
#
# hstack = np.hstack((baz,bingo))
# print(hstack)
#
# stack = np.stack((foo,bar),axis=0)
# print(stack)
# stack = np.stack((foo,bar),axis=1)
# print(stack)

############# SOrt prblm

# foo = np.array([1,7,43,127,13,62,1,3])
# print(np.sort(foo))
#
# boo = np.array([[55,2,12],[43,213,56],[543,63,123]])
# print(np.sort(boo))
# print(np.sort(boo,axis=0))
# print(np.sort(boo,axis=1))
#

# gar = np.array(["d","f","f","c"])
# print(np.unique(gar))

######### Problem souluton 10x2 Array movie
#
# genarator = np.random.default_rng(123)
# ratings = np.round(genarator.uniform(0.0,10.0,size=(10,2)))
# ratings[[1,2,7,9],[0,0,0,0]]=np.nan
# # print(ratings)
#
#
# x = np.where(np.isnan(ratings[:,0]),ratings[:,1],ratings[:,0])
#
# result = np.insert(arr=ratings,values=x,axis=1,obj=2)
# # print(result)



####### Fish problem
# locs = np.array([[0,0,0],[1,1,2],[0,0,0],[2,1,3],[5,5,4],[5,0,0],[0,0,0,],[2,1,3],[2,1,4],[1,2,3]])
# print(locs)
#
# genarator = np.random.default_rng(1010)
# weights = genarator.normal(loc=5,size=10)
#
# print(weights)
#
# sortedfish = np.argsort(weights)[::-1]
# print(sortedfish)
# uniques,first_idx = np.unique(locs[sortedfish],axis=0,return_index=True)
# print(uniques)
# print(first_idx)
# survivors = sortedfish[first_idx]
# print(survivors)
