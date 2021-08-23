
@author: mohammed batis 18511160002
"""

import numpy as np


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

    sortedClassCount = sorted(classCount.items(),\
                              key=lambda x:x[1], reverse=True)
    return sortedClassCount[0][0] 

def autoNorm(dataSet):# dataset 100 3
    minVals = dataSet.min(axis=0)
    maxVals = dataSet.max(axis=0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))##(100,3)
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet/np.tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

def split2(filename,sep=",",ratio=0.2,randsel=True):
    fr = open(filename)
    numberOfLines = len(fr.readlines())-1
    fr.seek(0,0)
    line = fr.readline()
    colnum = len(line.split(sep))
    
    returnMat = np.zeros((numberOfLines,colnum-1))
    classLabelVercot = []
    
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFormLine = line.split(sep)
       
        if(len(listFormLine) > 1):
            returnMat[index,:] = listFormLine [0:colnum-1]
            
        classLabelVercot.append(listFormLine[-1])
        index +=1
    classLabelVercot = np.array(classLabelVercot)
    normMat, ranges, minVals = autoNorm(returnMat)
    trainLines = numberOfLines - int(numberOfLines*ratio)

    if randsel:
        indexselect = CreateRandomList(numberOfLines,trainLines)
        Xtrain = normMat[indexselect]
        Ytrain = classLabelVercot[indexselect]
        Xtest = normMat[~indexselect]
        Ytest = classLabelVercot[~indexselect]
        
    else:
        Xtrain = normMat[0:trainLines,:]
        Ytrain = classLabelVercot[0:trainLines]
        Xtest = normMat[trainLines:,:]
        Ytest = classLabelVercot[trainLines:]
   
    return Xtrain, Ytrain,Xtest, Ytest;

def CreateRandomList(maxNum,num):
    a=[False]*maxNum
    while sum(a)<num:
    
        inx=np.random.randint(0,maxNum-1)
        a[inx]=True;
        
    return np.array(a)


def TestWine():
    Xtrain,Ytrain,Xtest,Ytest=split2("winequality.csv")
    len1=len(Xtest)
    countCorrect = 0
    countError = 0
    for i in range(len1):
        res=classify0(Xtest[i],Xtrain,Ytrain,k=6)
        print("real answer =", Ytest[i], "classifier answer=", res)
        if(res==Ytest[i]):
            countCorrect=countCorrect+1
        else:
            countError=countError+1
    print("Correct rate = ",countCorrect/len1)
    print("Error rate = ",countError/len1)
    print(len1)
    print(countCorrect)
    print(countError)

    
def TestIris():
    Xtrain,Ytrain,Xtest,Ytest=split2("irisdata.txt")
    len1=len(Xtrain)
    countCorrect = 0
    countError = 0
    for i in range(len1):
        res=classify0(Xtrain[i],Xtrain,Ytrain,k=6)
        print("real answer =", Ytrain[i], "classifier answer=", res)
        if(res==Ytrain[i]):
            countCorrect=countCorrect+1
        else:
            countError=countError+1
    print("Correct rate = ",countCorrect/len1)
    print("Error rate = ",countError/len1)
    print(len1)
    print(countCorrect)
    print(countError)
    
if __name__ == '__main__':
  #  TestWine()
    
     TestIris()
