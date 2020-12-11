import matplotlib.pyplot as plt
import csv
import os

directory = r'data/'
for filepath in os.listdir(directory):
    if filepath.endswith(".csv"): 
        x = []
        y = []
        with open(directory + filepath) as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            for row in plots:
                if row[0] == 'x':
                    continue
                x.append(float(row[0]))
                y.append(float(row[1]))
        plt.scatter(x,y, label='Mouse Location')
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.xlim(0,1)
        # plt.ylim(0,1)
        plt.title('GAN Mouse Trajectory')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.savefig(directory + filepath[:-4])
