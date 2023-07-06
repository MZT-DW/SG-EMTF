import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 13,
             }  ##设置图例大小位置
    x4=np.array([1,2,3,4, 5, 6, 7])
    x5 = np.array([1,2,3,4, 5, 6, 7, 8])# x4和x5用来调整柱子位置，和显示横坐标刻度
    y1=np.array([0.250, 1.184, 2.352, 4.651, 6.974,
9.336, 11.575])  #柱一数据
    y2 = np.array([0.618, 2.778, 5.650, 11.495, 19.553, 27.845, 36.186])  #柱二数据
    #y1=np.divide(y1, 1000)
    #y2=np.divide(y2,1000)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.ylabel(u'Average Iterime /s', font1)
    plt.xlabel(u'Task Num', font1)
    plt.bar(x4+0.3, y1, 0.35,color="b", align="center", label="SG-EMTF")#融合威胁评估方法
    #bar方法画柱状图，x4数组用来调整柱子的位置，x4+0.3代表第一个柱子的横坐标，y1为柱子的高度，即纵坐标
    plt.bar(x4+0.65, y2, 0.35,color="g", align="center", label="GEMTA")#TOPSIS方法
    # bar方法画柱状图，x4数组用来调整柱子的位置，x4+0.65代表第二个柱子的横坐标，y2为柱子的高度，即纵坐标

    plt.xticks(x5,[ '100', '500', '1000', '2000', '3000', '4000', '5000', '6000'])##横坐标刻度标签,因为我是把柱子画在刻度之间，所以画4个柱子要5个刻度
    #for i in range(0,2): #for内语句为添加柱状图的数据标签，x4[i]+0.3代表数据横坐标，y1[i]+0.0003代表数据纵坐标，y1[i]数据标签内容
    #  plt.text(x4[i]+0.3, y1[i]+1,y1[i], ha='center', va='bottom', fontsize=9,color = "b")
    #  plt.text(x4[i] + 0.65, y2[i] + 1, y2[i], ha='center', va='bottom', fontsize=9, color="g")
    for a, b in zip(x4, y1):
        plt.text(a + 0.3, b + 0.1, round(b,2), ha='center', va='bottom', fontsize=10)
    for a, b in zip(x4, y2):
        plt.text(a + 0.65, b + 0.1, round(b,2), ha='center', va='bottom', fontsize=10)
    plt.plot(x4+0.3, y1,'o--', color="b")
    plt.plot(x4+0.65, y2,'o--', color="g")
    plt.legend()

    plt.show()