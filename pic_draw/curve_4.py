import matplotlib.pyplot as plt
import numpy as np
x_1 = np.arange(10, 100, 10)
y1 = [
1.272,
1.538,
1.796,
2.122,
2.324,
2.870,
3.104,
3.261,
3.711,
3.715,

6.697,
9.829,
13.004,
18.352,
21.806,
25.591,
29.041,
34.375,
36.307,
40.628,
43.008,
48.311,
50.135,
54.654,
]
#x_2 = np.arange(10, 100, 10)
x_2 = np.append(np.arange(10, 100, 10), np.arange(100, 1600, 100))
print(x_2)
y2 = [
3.220,
3.961,
4.444,
4.848,
5.582,
6.158,
7.807,
8.577,
9.338,
]
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 13,
         }##设置图例大小位置
#fig = plt.figure(figsize=(8,5))
fig, (ax1, ax2)=plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 10]}, sharey=True, dpi=100)

ax1.plot(x_2, y1, label='GEMTA')
ax1.plot(x_1, y2, label='SG-EMTF')
ax2.plot(x_2, y1)
ax1.set_xlim(0, 100)
ax2.set_xlim(100, 1600)
ax2.yaxis.tick_right()
d= .65
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10,
              linestyle='none', color='k', mew=1, clip_on=False)

ax1.plot([1, 1], [1, 0],transform=ax1.transAxes, **kwargs)
ax1.set_xticks(range(0, 100, 45))
#ax2.set_xticks(range(200, 1600, 100))
plt.tight_layout()
#ax1.grid(axis='y')
#ax2.grid()
ax1.spines['right'].set_visible(False)#关闭子图1中底部脊
ax2.spines['left'].set_visible(False)##关闭子图2中顶部脊
ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)
#ax2.grid()
plt.xlabel('DIMENSION', fontdict=font1)
ax1.set_ylabel('Average Itertime/s', fontdict=font1)
plt.tight_layout()



lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

plt.show()
'''
bax.plot(x_1, y1, label='GPU-based')
bax.plot(x_1, y2, label='GEMTA')
bax.legend(loc='upper left')
bax.set_xlabel('DIMENSION', fontdict=font1)
bax.set_ylabel('Iteration Time /s', fontdict=font1)
plt.show()
'''