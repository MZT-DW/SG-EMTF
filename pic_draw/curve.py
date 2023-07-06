import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
plt.style.use('seaborn-whitegrid')
palette = pyplot.get_cmap('Set1')
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size'  : 12,}

fig = plt.figure(figsize=(15,9))
iters = list(range(0, 1001, 50))

alldata1 = np.array([
110800.625000,
82926.984375,
75298.515625,
70021.625000,
68426.218750,
66205.210938,
64042.710938,
62841.105469,
61364.058594,
60960.777344,
60309.855469,
58880.199219,
57391.394531,
57267.527344,
56870.859375,
56049.660156,
55515.816406,
55096.242188,
54198.042969,
53845.113281,
53569.171875,
])

alldata2 = np.array([
106292.757813,
81104.156250,
72094.398438,
69054.968750,
66645.710938,
63992.410156,
61952.492188,
60980.070313,
59171.074219,
58708.316406,
58066.609375,
57566.609375,
55994.277344,
55420.710938,
55184.046875,
55021.765625,
54601.855469,
54214.621094,
53743.617188,
53282.710938,
53132.468750,
])

alldata3 = np.array([
114333.875000,
86241.882813,
77678.703125,
74476.851563,
70693.156250,
69279.828125,
67086.125000,
65308.535156,
64243.066406,
63033.164063,
62107.257813,
61568.324219,
60930.019531,
60484.964844,
59948.574219,
59261.621094,
58394.890625,
57732.703125,
57282.488281,
56963.750000,
56855.695313,
])

allstd1 = np.array([
3032.740234,
6454.613770,
3488.483398,
3679.910156,
3981.119873,
4200.675293,
3230.328369,
2933.353271,
2987.315918,
5043.677734,
4449.295410,
4367.893066,
3614.430176,
3614.430176,
3718.792969,
3649.772949,
3949.972412,
3555.561279,
3508.568115,
3843.992920,
3843.992920,
])

allstd2 = np.array([
4395.095703,
5129.040039,
2543.765137,
2885.519043,
3113.395508,
3463.047119,
2517.154785,
1852.708130,
2727.087646,
2686.312500,
3742.899658,
4459.433594,
4350.806641,
3812.981445,
3691.916504,
3055.332520,
2732.803955,
2663.849121,
2502.562988,
2811.769531,
2811.769531,
])

allstd3 = np.array([
5597.804688,
5256.738770,
5217.755859,
3976.184570,
3282.546631,
4344.609863,
4394.619629,
4430.337891,
4299.622559,
4145.005859,
4488.673828,
4386.934570,
3620.806396,
3381.336426,
3324.296143,
3146.275635,
2977.605225,
2983.006836,
2930.645752,
2914.052734,
2857.027832,
])

color = palette(0)
ax_0 = fig.add_subplot(3,3,1)

r1 = list(map(lambda x: x[0] - x[1], zip(alldata1, allstd1)))
r2 = list(map(lambda x: x[0] + x[1], zip(alldata1, allstd1)))
ax_0.plot(iters, alldata1, color=color, label="CPU-based", linewidth=3.0)
ax_0.fill_between(iters, r1, r2, color=color, alpha=0.2)

color=palette(1)
r1 = list(map(lambda x: x[0] - x[1], zip(alldata2, allstd2)))
r2 = list(map(lambda x: x[0] + x[1], zip(alldata2, allstd2)))
ax_0.plot(iters, alldata2, color=color, label="SG-EMTF", linewidth=3.0)
ax_0.fill_between(iters, r1, r2, color=color, alpha=0.2)

color=palette(2)
r1 = list(map(lambda x: x[0] - x[1], zip(alldata3, allstd3)))
r2 = list(map(lambda x: x[0] + x[1], zip(alldata3, allstd3)))
ax_0.plot(iters, alldata3, color=color, label="I-GEMTA", linewidth=3.0)
ax_0.fill_between(iters, r1, r2, color=color, alpha=0.2)

ax_0.set_title(u'Sphere')
ax_0.legend(loc='upper right', prop=font1)
ax_0.set_xlabel('Iteration', fontsize=12)
ax_0.set_ylabel('Fitness', fontsize=12)

plt.xticks(size=12)
plt.yticks(size=12)

alldata1 = np.array([
3172539136.000000,
1928185728.000000,
1693729536.000000,
1450495232.000000,
1338945536.000000,
1262123008.000000,
1210793856.000000,
1194645376.000000,
1157958912.000000,
1140867200.000000,
1114793600.000000,
1088041600.000000,
1059265664.000000,
1039794560.000000,
1010111488.000000,
1004491136.000000,
989078400.000000,
967200512.000000,
954604608.000000,
933805824.000000,
914234944.000000,
])

alldata2 = np.array([
3032668928.000000,
1965081984.000000,
1634322432.000000,
1444801280.000000,
1386509824.000000,
1276983296.000000,
1196163840.000000,
1124377216.000000,
1100890752.000000,
1073351040.000000,
1043060224.000000,
1033767232.000000,
999780224.000000,
984125888.000000,
942684800.000000,
926798144.000000,
909208448.000000,
902977856.000000,
893596160.000000,
861660800.000000,
848175808.000000,
])


alldata3 = np.array([
3278992128.000000,
2182828544.000000,
1860169088.000000,
1679472768.000000,
1569529984.000000,
1467994624.000000,
1357829888.000000,
1308014848.000000,
1282540288.000000,
1246580992.000000,
1195012992.000000,
1184676096.000000,
1121891712.000000,
1108893568.000000,
1056303424.000000,
1042923072.000000,
1022687040.000000,
1005373248.000000,
996168512.000000,
996168512.000000,
983062016.000000,
])

allstd1 = np.array([
165908080.000000,
290984192.000000,
242260368.000000,
157658144.000000,
237772576.000000,
223418448.000000,
207353280.000000,
175198048.000000,
161459552.000000,
113119656.000000,
114407024.000000,
95206376.000000,
111956952.000000,
111956952.000000,
131189624.000000,
126657352.000000,
114949544.000000,
114949544.000000,
114576056.000000,
111144480.000000,
97944472.000000,
])

allstd2 = np.array([
320839168.000000,
123664792.000000,
161576752.000000,
114246352.000000,
163275568.000000,
160747504.000000,
135659152.000000,
154740512.000000,
106685264.000000,
125574392.000000,
87562968.000000,
86386360.000000,
86386360.000000,
79885168.000000,
75455984.000000,
72354392.000000,
83897064.000000,
78241200.000000,
85466040.000000,
84229528.000000,
84229528.000000,
])

allstd3 = np.array([
423702912.000000,
120173128.000000,
172001744.000000,
161076768.000000,
132856272.000000,
158058496.000000,
150283552.000000,
140673776.000000,
73929896.000000,
73929896.000000,
68800024.000000,
86225216.000000,
86225216.000000,
101584160.000000,
73083552.000000,
82851248.000000,
85074720.000000,
75238864.000000,
75238864.000000,
75238864.000000,
93746352.000000,
])

color = palette(0)
ax_1 = fig.add_subplot(3,3,2)

r1 = list(map(lambda x: x[0] - x[1], zip(alldata1, allstd1)))
r2 = list(map(lambda x: x[0] + x[1], zip(alldata1, allstd1)))
ax_1.plot(iters, alldata1, color=color, label="CPU-based", linewidth=3.0)
ax_1.fill_between(iters, r1, r2, color=color, alpha=0.2)

color=palette(1)
r1 = list(map(lambda x: x[0] - x[1], zip(alldata2, allstd2)))
r2 = list(map(lambda x: x[0] + x[1], zip(alldata2, allstd2)))
ax_1.plot(iters, alldata2, color=color, label="SG-EMTF", linewidth=3.0)
ax_1.fill_between(iters, r1, r2, color=color, alpha=0.2)

color=palette(2)
r1 = list(map(lambda x: x[0] - x[1], zip(alldata3, allstd3)))
r2 = list(map(lambda x: x[0] + x[1], zip(alldata3, allstd3)))
ax_1.plot(iters, alldata3, color=color, label="I-GEMTA", linewidth=3.0)
ax_1.fill_between(iters, r1, r2, color=color, alpha=0.2)

ax_1.set_title(u'Rosenbrock')
ax_1.legend(loc='upper right', prop=font1)
ax_1.set_xlabel('Iteration', fontsize=12)
ax_1.set_ylabel('Fitness', fontsize=12)

plt.xticks(size=12)
plt.yticks(size=12)

alldata1 = np.array([
21.321810,
21.138525,
21.106718,
21.085649,
21.074461,
21.060406,
21.052059,
21.050936,
21.046835,
21.042103,
21.040144,
21.036757,
21.034084,
21.029562,
21.029488,
21.028923,
21.020727,
21.020727,
21.020424,
21.018484,
21.017866,

])

alldata2 = np.array([
21.301632,
21.094889,
21.067528,
21.051723,
21.039782,
21.027840,
21.024935,
21.017023,
21.016045,
21.010788,
21.009888,
21.008486,
21.004538,
21.000639,
20.998238,
20.996557,
20.993729,
20.993652,
20.993652,
20.991383,
20.991131,
])

alldata3 = np.array([
21.349022,
21.137791,
21.113037,
21.098061,
21.083290,
21.082180,
21.074486,
21.066713,
21.062134,
21.055115,
21.051731,
21.050528,
21.049757,
21.047909,
21.044043,
21.037010,
21.035494,
21.024521,
21.024105,
21.020626,
21.020256,
])

allstd1 = np.array([
0.074218,
0.024794,
0.025527,
0.026277,
0.038608,
0.053062,
0.045930,
0.045930,
0.045930,
0.051838,
0.051838,
0.050348,
0.050348,
0.046873,
0.046873,
0.046873,
0.044695,
0.043657,
0.043657,
0.043657,
0.043657,
])

allstd2 = np.array([
0.042094,
0.026735,
0.042579,
0.040829,
0.040951,
0.037350,
0.037039,
0.036855,
0.030280,
0.036775,
0.037819,
0.037819,
0.037819,
0.034793,
0.034793,
0.037771,
0.037771,
0.037836,
0.035469,
0.035177,
0.035177,
])

allstd3 = np.array([
0.058164,
0.037642,
0.041315,
0.041037,
0.039670,
0.041076,
0.038084,
0.041673,
0.036012,
0.035613,
0.034766,
0.032417,
0.034636,
0.042796,
0.038518,
0.039398,
0.054406,
0.070854,
0.071198,
0.072577,
0.072131,
])

color = palette(0)
ax_2 = fig.add_subplot(3,3,3)

r1 = list(map(lambda x: x[0] - x[1], zip(alldata1, allstd1)))
r2 = list(map(lambda x: x[0] + x[1], zip(alldata1, allstd1)))
ax_2.plot(iters, alldata1, color=color, label="CPU-based", linewidth=3.0)
ax_2.fill_between(iters, r1, r2, color=color, alpha=0.2)

color=palette(1)
r1 = list(map(lambda x: x[0] - x[1], zip(alldata2, allstd2)))
r2 = list(map(lambda x: x[0] + x[1], zip(alldata2, allstd2)))
ax_2.plot(iters, alldata2, color=color, label="SG-EMTF", linewidth=3.0)
ax_2.fill_between(iters, r1, r2, color=color, alpha=0.2)

color=palette(2)
r1 = list(map(lambda x: x[0] - x[1], zip(alldata3, allstd3)))
r2 = list(map(lambda x: x[0] + x[1], zip(alldata3, allstd3)))
ax_2.plot(iters, alldata3, color=color, label="I-GEMTA", linewidth=3.0)
ax_2.fill_between(iters, r1, r2, color=color, alpha=0.2)

ax_2.set_title(u'Ackley')
ax_2.legend(loc='upper right', prop=font1)
ax_2.set_xlabel('Iteration', fontsize=12)
ax_2.set_ylabel('Fitness', fontsize=12)

plt.xticks(size=12)
plt.yticks(size=12)

alldata1 = np.array([
27594.488281,
20561.609375,
18682.470703,
17698.787109,
16915.394531,
16711.781250,
16338.887695,
16085.666016,
15828.870117,
15700.662109,
15475.490234,
15230.185547,
15076.964844,
15000.576172,
14821.847656,
14794.170898,
14750.126953,
14633.235352,
14522.845703,
14438.745117,
14302.176758,
])

alldata2 = np.array([
27200.970703,
20461.771484,
18754.216797,
17467.451172,
16864.287109,
16548.935547,
16217.426758,
15951.302734,
15544.373047,
15331.283203,
14895.767578,
14817.482422,
14509.757813,
14426.255859,
14233.449219,
14150.814453,
14011.588867,
13930.980469,
13832.784180,
13775.728516,
13551.128906,
])

alldata3 = np.array([
29525.955078,
22180.511719,
19857.097656,
18973.001953,
18008.607422,
17177.074219,
17081.042969,
16651.373047,
16315.020508,
15874.567383,
15846.132813,
15815.382813,
15631.889648,
15529.629883,
15428.684570,
15155.534180,
14895.745117,
14870.726563,
14715.601563,
14601.050781,
14500.406250,
])

allstd1 = np.array([
2330.089111,
1443.279541,
1859.968628,
1554.561523,
1184.185425,
994.902161,
957.845520,
1013.860107,
1133.952271,
983.905640,
1137.929565,
1066.166382,
1051.286377,
889.881958,
861.859863,
878.187317,
878.187317,
878.187317,
1140.947998,
1140.947998,
1140.947998,
])

allstd2 = np.array([
1191.945435,
1730.883789,
1071.991333,
1046.372925,
846.210938,
765.184998,
769.346558,
711.872559,
719.654785,
875.232117,
914.513245,
794.478149,
798.156250,
723.321167,
716.998047,
942.005981,
941.995178,
935.489990,
1044.515869,
972.417725,
935.864685,
])

allstd3 = np.array([
1390.207642,
1237.839966,
1382.174561,
838.704529,
870.358337,
1002.930725,
890.045776,
872.400757,
1350.919800,
1176.541748,
1148.441895,
1130.463501,
1150.984497,
1109.736084,
1067.308716,
953.405762,
943.917847,
1029.707397,
1016.621033,
998.813904,
1007.444885,
])

color = palette(0)
ax_3 = fig.add_subplot(3,3,4)

r1 = list(map(lambda x: x[0] - x[1], zip(alldata1, allstd1)))
r2 = list(map(lambda x: x[0] + x[1], zip(alldata1, allstd1)))
ax_3.plot(iters, alldata1, color=color, label="CPU-based", linewidth=3.0)
ax_3.fill_between(iters, r1, r2, color=color, alpha=0.2)

color=palette(1)
r1 = list(map(lambda x: x[0] - x[1], zip(alldata2, allstd2)))
r2 = list(map(lambda x: x[0] + x[1], zip(alldata2, allstd2)))
ax_3.plot(iters, alldata2, color=color, label="SG-EMTF", linewidth=3.0)
ax_3.fill_between(iters, r1, r2, color=color, alpha=0.2)

color=palette(2)
r1 = list(map(lambda x: x[0] - x[1], zip(alldata3, allstd3)))
r2 = list(map(lambda x: x[0] + x[1], zip(alldata3, allstd3)))
ax_3.plot(iters, alldata3, color=color, label="I-GEMTA", linewidth=3.0)
ax_3.fill_between(iters, r1, r2, color=color, alpha=0.2)

ax_3.set_title(u'Rastrgin')
ax_3.legend(loc='upper right', prop=font1)
ax_3.set_xlabel('Iteration', fontsize=12)
ax_3.set_ylabel('Fitness', fontsize=12)

plt.xticks(size=12)
plt.yticks(size=12)

alldata1 = np.array([
28.404985,
21.004204,
19.512627,
18.464375,
17.768278,
17.679478,
17.218542,
16.839422,
16.597307,
16.289492,
16.136305,
15.972407,
15.815329,
15.644629,
15.550610,
15.388386,
15.282486,
15.137441,
15.037911,
15.004794,
14.737780,
])

alldata2 = np.array([
27.335218,
21.069040,
19.405144,
18.081743,
17.212021,
16.736095,
16.260944,
15.879369,
15.770648,
15.599174,
15.434799,
15.144559,
15.091049,
14.927648,
14.793877,
14.682423,
14.537750,
14.433949,
14.395178,
14.318459,
14.189216,
])

alldata3 = np.array([
29.137331,
22.113550,
20.233480,
19.319639,
18.788504,
18.424168,
18.008625,
17.628721,
17.254202,
16.798508,
16.650230,
16.393042,
16.375292,
16.229519,
15.986588,
15.758901,
15.732617,
15.494267,
15.375530,
15.110979,
15.066345,
])

allstd3 = np.array([
1.622568,
1.521311,
1.379789,
1.540608,
1.289690,
1.035333,
0.810175,
0.653255,
0.569935,
0.663338,
0.592748,
0.615623,
0.539546,
0.556428,
0.661594,
0.621661,
0.586444,
0.692285,
0.681922,
0.695898,
0.767209,
])

allstd1 = np.array([
1.793926,
1.340675,
1.011526,
0.931612,
0.989239,
0.682553,
0.728727,
0.499983,
0.759031,
0.756571,
1.041353,
0.898460,
0.898460,
0.833032,
0.833032,
0.833032,
0.773459,
0.700507,
0.700507,
0.700507,
0.687124,
])

allstd2 = np.array([
1.982637,
1.334609,
1.006097,
0.818941,
1.018100,
0.775437,
0.958690,
0.958794,
0.980229,
0.959714,
0.855625,
0.836243,
0.665004,
0.576313,
0.781515,
0.714820,
0.714820,
0.714820,
0.736778,
0.736778,
0.717889,
])

color = palette(0)
ax_4 = fig.add_subplot(3,3,5)

r1 = list(map(lambda x: x[0] - x[1], zip(alldata1, allstd1)))
r2 = list(map(lambda x: x[0] + x[1], zip(alldata1, allstd1)))
ax_4.plot(iters, alldata1, color=color, label="CPU-based", linewidth=3.0)
ax_4.fill_between(iters, r1, r2, color=color, alpha=0.2)

color=palette(1)
r1 = list(map(lambda x: x[0] - x[1], zip(alldata2, allstd2)))
r2 = list(map(lambda x: x[0] + x[1], zip(alldata2, allstd2)))
ax_4.plot(iters, alldata2, color=color, label="SG-EMTF", linewidth=3.0)
ax_4.fill_between(iters, r1, r2, color=color, alpha=0.2)

color=palette(2)
r1 = list(map(lambda x: x[0] - x[1], zip(alldata3, allstd3)))
r2 = list(map(lambda x: x[0] + x[1], zip(alldata3, allstd3)))
ax_4.plot(iters, alldata3, color=color, label="I-GEMTA", linewidth=3.0)
ax_4.fill_between(iters, r1, r2, color=color, alpha=0.2)

ax_4.set_title(u'Griewank')
ax_4.legend(loc='upper right', prop=font1)
ax_4.set_xlabel('Iteration', fontsize=12)
ax_4.set_ylabel('Fitness', fontsize=12)

plt.xticks(size=12)
plt.yticks(size=12)

alldata1 = np.array([
83.166222,
75.824669,
73.136429,
71.832848,
70.689819,
70.308350,
69.738426,
69.395142,
69.124176,
68.927017,
68.512848,
68.389397,
68.253464,
68.045822,
67.962212,
67.926315,
67.790726,
67.637115,
67.507835,
67.442375,
67.351311,
])

alldata2 = np.array([
81.861488,
75.011246,
73.212425,
71.800644,
70.696114,
70.129013,
69.607109,
68.858307,
68.692497,
68.063362,
67.760040,
67.576897,
67.532906,
67.395309,
67.383629,
67.161232,
66.893799,
66.791389,
66.531837,
66.482887,
66.355904,
])


alldata3 = np.array([
85.314301,
76.971230,
74.633377,
73.314011,
72.073715,
71.279091,
70.443832,
69.965843,
69.544273,
69.445648,
69.210335,
68.934616,
68.840485,
68.788712,
68.546501,
68.470940,
68.470940,
68.261185,
67.823273,
67.751610,
67.655212,
])

allstd3 = np.array([
3.147657,
1.774086,
1.374138,
1.561639,
1.660619,
1.464618,
1.221697,
1.039257,
1.097925,
1.766811,
1.787812,
1.680644,
1.666980,
1.639882,
1.615436,
1.607535,
1.674458,
1.594966,
1.600179,
1.551839,
1.401631,
])

allstd1 = np.array([
2.574328,
1.342812,
1.473390,
1.218247,
1.315096,
1.272686,
1.559021,
1.555208,
1.429627,
1.220599,
1.131971,
1.012320,
1.040271,
0.925210,
0.924996,
0.820949,
0.820949,
1.255875,
1.241915,
1.223062,
1.303928,
])

allstd2 = np.array([
1.285360,
2.106140,
1.068308,
0.951088,
1.140949,
1.284547,
0.774464,
1.265793,
1.268145,
1.395702,
1.361014,
1.373027,
1.169995,
1.169995,
1.061397,
1.000166,
1.036999,
1.071949,
1.042061,
1.274283,
1.274283,
])

color = palette(0)
ax_5 = fig.add_subplot(3,3,6)

r1 = list(map(lambda x: x[0] - x[1], zip(alldata1, allstd1)))
r2 = list(map(lambda x: x[0] + x[1], zip(alldata1, allstd1)))
ax_5.plot(iters, alldata1, color=color, label="CPU-based", linewidth=3.0)
ax_5.fill_between(iters, r1, r2, color=color, alpha=0.2)

color=palette(1)
r1 = list(map(lambda x: x[0] - x[1], zip(alldata2, allstd2)))
r2 = list(map(lambda x: x[0] + x[1], zip(alldata2, allstd2)))
ax_5.plot(iters, alldata2, color=color, label="SG-EMTF", linewidth=3.0)
ax_5.fill_between(iters, r1, r2, color=color, alpha=0.2)

color=palette(2)
r1 = list(map(lambda x: x[0] - x[1], zip(alldata3, allstd3)))
r2 = list(map(lambda x: x[0] + x[1], zip(alldata3, allstd3)))
ax_5.plot(iters, alldata3, color=color, label="I-GEMTA", linewidth=3.0)
ax_5.fill_between(iters, r1, r2, color=color, alpha=0.2)

ax_5.set_title(u'Weierstrass')
ax_5.legend(loc='upper right', prop=font1)
ax_5.set_xlabel('Iteration', fontsize=12)
ax_5.set_ylabel('Fitness', fontsize=12)
plt.xticks(size=12)
plt.yticks(size=12)

alldata1 = np.array([
16998.126953,
15704.703125,
15330.090820,
15007.468750,
14916.211914,
14820.026367,
14725.636719,
14605.545898,
14594.653320,
14578.332031,
14546.118164,
14502.086914,
14480.178711,
14480.178711,
14461.330078,
14445.316406,
14420.091797,
14414.575195,
14391.672852,
14338.829102,
14329.068359,
])

alldata2 = np.array([
16734.007813,
14942.283203,
14650.238281,
14315.256836,
14175.940430,
14124.465820,
14105.524414,
14098.950195,
14088.237305,
13979.140625,
13918.455078,
13874.296875,
13774.050781,
13736.254883,
13713.930664,
13667.913086,
13659.894531,
13658.792969,
13560.652344,
13510.745117,
13498.692383,
])

alldata3 = np.array([
17441.441406,
15397.596680,
15075.044922,
14773.505859,
14676.348633,
14568.822266,
14485.415039,
14370.623047,
14242.067383,
14168.815430,
14139.398438,
14053.396484,
14012.153320,
13962.110352,
13926.763672,
13915.131836,
13871.271484,
13835.415039,
13833.804688,
13820.123047,
13775.960938,
])

allstd3 = np.array([
472.873383,
557.950073,
458.822357,
422.487762,
539.733826,
524.331055,
486.638000,
528.581055,
536.047058,
536.047058,
525.790710,
464.059204,
464.059204,
464.137329,
483.520386,
465.045715,
465.045715,
426.095306,
328.473694,
328.473694,
325.389984,
])

allstd1 = np.array([
583.825745,
286.839233,
295.624054,
292.224762,
322.894501,
321.341156,
319.744537,
252.518356,
275.447601,
268.178711,
200.783615,
192.293259,
176.013641,
223.751862,
223.751862,
204.824173,
195.282211,
191.850220,
191.850220,
212.936600,
212.936600,
])

allstd2 = np.array([
292.150421,
277.549286,
369.393402,
262.264008,
244.190018,
153.236740,
135.559143,
233.722046,
214.938156,
210.359711,
229.383835,
229.383835,
229.383835,
195.877060,
197.605484,
204.704498,
191.719376,
191.719376,
194.151230,
223.246582,
292.006042,
])

color = palette(0)
ax_6 = fig.add_subplot(3,3,7)

r1 = list(map(lambda x: x[0] - x[1], zip(alldata1, allstd1)))
r2 = list(map(lambda x: x[0] + x[1], zip(alldata1, allstd1)))
ax_6.plot(iters, alldata1, color=color, label="CPU-based", linewidth=3.0)
ax_6.fill_between(iters, r1, r2, color=color, alpha=0.2)

color=palette(1)
r1 = list(map(lambda x: x[0] - x[1], zip(alldata2, allstd2)))
r2 = list(map(lambda x: x[0] + x[1], zip(alldata2, allstd2)))
ax_6.plot(iters, alldata2, color=color, label="SG-EMTF", linewidth=3.0)
ax_6.fill_between(iters, r1, r2, color=color, alpha=0.2)

color=palette(2)
r1 = list(map(lambda x: x[0] - x[1], zip(alldata3, allstd3)))
r2 = list(map(lambda x: x[0] + x[1], zip(alldata3, allstd3)))
ax_6.plot(iters, alldata3, color=color, label="I-GEMTA", linewidth=3.0)
ax_6.fill_between(iters, r1, r2, color=color, alpha=0.2)

ax_6.set_title(u'Schwefel')
ax_6.legend(loc='upper right', prop=font1)
ax_6.set_xlabel('Iteration', fontsize=12)
ax_6.set_ylabel('Fitness', fontsize=12)
plt.xticks(size=12)
plt.yticks(size=12)
plt.subplots_adjust(left=0.06, right=0.98, bottom=0.05, top=0.94, wspace=0.20, hspace=0.27)
plt.show()