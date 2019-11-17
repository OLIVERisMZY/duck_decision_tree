import math
from collections import Counter
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

df = pd.read_csv('D:\可爱臭鸭鸭\duck_data.csv')#需要添加目标文件目录
df=df.drop(['food','health'],axis=1)
#========================
train_property=[]
for key in df.keys():
    train_property.append(key)
train_property.remove('index')
train_property.remove('good duck')
#print('鸭鸭属性集为：')
#print(train_property)
prototype_property=train_property[:]
#========================
train_data = np.array(df.iloc[:,0:8])

#print('一只鸭鸭的数据为：')
#print(train_data[0])
#========================
label=train_data[:,7]
count=len(train_data)
#print('全部鸭鸭的标签为：')
#print(label)
#======计算信息熵================
def Entropy(dataSet):
    dcount=len(dataSet)
    value=['yes','no']#记录值的名称
    value_count=[]
    for v in value:
       num=0
       for i in range(len(dataSet)):
           if dataSet[i,-1]==v:
               num+=1
       value_count.append(num)
    shannonEnt = 0.0  # 初始化信息熵
    for num in value_count:
        prob = float(num)/float(dcount)
        if prob != 0:
          shannonEnt -= prob *math.log(prob,2) #log base 2  计算信息熵
    return shannonEnt

def split_dataset(dataset,property,value):
    axis=0
    for i in range(len(prototype_property)):
        if prototype_property[i]==property:
            axis=i+1
            break
    ret_data=[]
    for j in range(len(dataset)):
        if dataset[j,axis]==value:
            ret_data.append(dataset[j])
    ret_data=np.array(ret_data)
    return ret_data

EntD=Entropy(train_data)


def get_Gain(dataset,sub_property):
  dcount=len(dataset)

  Gain = []
  for j in range(len(sub_property)):
    for k in range(len(prototype_property)):
        if prototype_property[k]==sub_property[j]:
            axis=k+1
            break
    Ga=EntD
    value=[]
    value_count=[]
    for i in range(dcount):
        if dataset[i,axis] not in value:
            value.append(dataset[i,axis])
    for va in value:
        m=0
        for i in range(dcount):
            if dataset[i,axis]==va:
                m+=1
        value_count.append(m)
    for k in range(len(value)):
        ra_data=split_dataset(dataset,sub_property[j],value[k])
        ent=Entropy(ra_data)
        Ga-=ent*float(value_count[k])/dcount
    Gain.append(Ga)
  return  Gain


def choose_best_pro(dataset,sub_property):
    all_gain=get_Gain(dataset,sub_property)
    best_index=np.argmax(all_gain)
    return  best_index



def createTree(dataSet,property):
    classList = dataSet[:,7]#标签
    if len(np.unique(classList))==1:
        if classList[0]=='yes':
          return 'good duck'
        else:
          return 'big bad duck'
    if len(dataSet[0]) ==3:#标签，序号不计
        yes_num=0
        no_num=0
        for i in range(len(dataSet)):
            if dataSet[i,-1]=='yes':
                yes_num+=1
            else:
                no_num+=1
        if yes_num>=no_num:
          return '好鸭'
        else:
          return '坏鸭'
    best_index = choose_best_pro(dataSet,property)
    bestFeatLabel = property[best_index]
    myTree = {bestFeatLabel:{}}
    del(property[best_index])
    #print(property)
    axis=0
    for k in range(len(prototype_property)):
        if prototype_property[k]==bestFeatLabel:
            axis=k+1
            break
    featValues = np.unique(dataSet[:, axis])
    #print(featValues)
    for value in featValues:
        subLabels = property[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(split_dataset(dataSet, bestFeatLabel, value),subLabels)
    return myTree
tree=createTree(train_data,train_property)

print(tree)
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# 计算树的叶子节点数量
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


# 计算树的最大深度
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# 画节点
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


# 画箭头上的文字
def plotMidText(cntrPt, parentPt, txtString):
    lens = len(txtString)
    xMid = (parentPt[0] + cntrPt[0]) / 2.0 - lens * 0.002
    yMid = (parentPt[1] + cntrPt[1]) / 2.0
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.x0ff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.y0ff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.y0ff = plotTree.y0ff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.x0ff = plotTree.x0ff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.x0ff, plotTree.y0ff), cntrPt, leafNode)
            plotMidText((plotTree.x0ff, plotTree.y0ff), cntrPt, str(key))
    plotTree.y0ff = plotTree.y0ff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.x0ff = -0.5 / plotTree.totalW
    plotTree.y0ff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

createPlot(tree)