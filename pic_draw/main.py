# This is a sample Python script.
import matplotlib.pyplot as plt
import matplotlib.font_manager
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    font={'family': 'Nimbus Roman',
     'weight': 'normal',
    'style' : 'normal',
      'size': 27
          }
    figsize = 10,8
    figure, ax = plt.subplots(figsize=figsize)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    #k1 = [13.91, 10.08, 8.24, 6.51, 5.33, 4.64, 4.28, 4.00, 3.65]
    #k2 = [20.175, 14.1835,10.412,9.8765,7.66567,6.241, 5.59, 5.07, 5.00]
    #k3_备 = [18.20, 13.28, 13.30, 11.84, 8.68, 6.38, 5.59, 5.07, 5.00]
    #k3_备2 = [16.404, 14.5186, 12.7304, 9.857, 8.2468, 7.2062, 5.4132, 5.3222, 4.755]
    #k3 = [17.9242, 15.7356, 12.3164, 11.8838, 10.1204, 8.6534, 8.018, 6.7358, 5.081, ]
    #k4 = [18.7075, 14.2445, 10.4655, 8.6335, 8.1145, 7.7775, 6.8355, 6.3065, 5.708]
    #k5 = [14.732, 11.6184, 9.2644, 7.93, 6.338, 5.41457, 5.01, 4.524, 4.362]


    k1 = [20.79, 15.56, 12.65, 9.92, 9.55, 8.62, 8.01, 7.42, 6.40]
    k2 = [27.643, 24.5345, 17.672, 15.228, 12.845, 11.8975, 9.9975, 9.679, 9.027]
    #k3_备 = [29.23, 24.60, 21.43, 17.07, 11.58, 10.21, 9.40, 8.55, 7.81]
    #k3_备2 = [33.7646, 24.6528, 19.0382, 18.9638, 17.3156, 15.6178, 10.9432, 10.0378, 9.3296, ]
    k3 = [31.4554,  27.1314, 18.6138, 17.827, 15.6966, 12.8556, 10.2888, 9.1525, 7.9924]
    k4 = [26.1155, 18.2565, 15.537, 13.5675, 12.181, 10.7595,10.2685, 9.8807, 9.7431]
    k5 = [22.523, 17.2766, 13.9986, 11.3102, 10.4964, 9.7686, 8.902, 8.2792, 7.5535]


    #k1 = [13.40, 9.97, 7.63, 6.54, 6.03, 4.99, 4.60, 4.21, 4.23]
    #k2 = [20.8236, 13.1498, 11.6666, 10.38, 9.1778, 7.724, 6.7242, 6.686, 5.8808,]
    #k3 = [17.36, 15.34, 12.71, 10.67, 8.88, 7.57, 6.73, 6.24, 6.08,]
    #k4 = [17.0775, 14.1665, 12.54, 11.133, 10.204, 8.471, 8.863, 7.779, 7.7555, ]
    #k4_备 = [19.3514, 15.7539, 13.1642, 9.7386, 10.759, 9.7142, 11.105, 8.7566, 8.4078, ]
    #k5 = [17.7096, 13.7076, 12.0088, 9.204, 7.5964, 6.222, 5.961, 5.361, 5.2846]
    A, = plt.plot(x, k1, 's-', color = 'r', label="SMOD-DE", linewidth = 3, ms = 12)
    B, = plt.plot(x, k2, 'o-', color = 'g', label="HotStar", linewidth = 3, ms = 12)
    C, = plt.plot(x, k3, 'v-', color = 'k', label="Density-based", linewidth = 3, ms = 10)
    D, = plt.plot(x, k4, 'D-', color = 'b', label="BAPS", linewidth = 3, ms = 10)
    E, = plt.plot(x, k5, 'X-', color = 'darkred', label="ORA-CC", linewidth = 3, ms = 10)

    bwith = 3
    bx = plt.gca()
    bx.spines['bottom'].set_linewidth(bwith)
    bx.spines['left'].set_linewidth(bwith)
    bx.spines['top'].set_linewidth(bwith)
    bx.spines['right'].set_linewidth(bwith)

    plt.rc('legend', fontsize=24)
    #legend = plt.legend(handles=[A, B, C], fontsize = 100)
    plt.tick_params(labelsize=24)
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontweight('bold') for label in labels]
    plt.xlabel("K", font)
    plt.xticks(range(1, 10, 2));
    plt.ylabel("AVERAGE RESPONSE TIME", font)
    plt.legend(loc = "best")
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
