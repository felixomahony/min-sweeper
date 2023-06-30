'''
Author: Felix O'Mahony
Date: 29/06/2023
License: See license.txt file
-----------
Description: this file generates data for the game. There are different requirements for each of the five levels, and the data separations get more difficult with each level.
'''
import numpy as np
import matplotlib.pyplot as plt
import json

n = 40
def generate_data(level):
    # 2 classes, 2 feature, n samples of each class
    class_0 = np.zeros((n,2))
    class_1 = np.zeros((n,2))
    if level == 0:
        # Generate data for level 0
        class_0[:,0] = np.random.uniform(-1, 1, size = (n))
        class_0[:,1] = np.random.uniform(0.2, 1, size = (n))
        class_1[:,0] = np.random.uniform(-1, 1, size = (n))
        class_1[:,1] = np.random.uniform(-1, -0.1, size = (n))
    elif level == 1:
        # Generate data for level 1
        class_0[:,0] = np.random.uniform(-1, 1, size = (n))
        class_0[:,1] = np.random.uniform(-0.1, 1, size = (n))
        class_1[:,0] = np.random.uniform(-1, 1, size = (n))
        class_1[:,1] = np.random.uniform(-1, 0.2, size = (n))
    elif level == 2:
        # Generate data for level 2
        class_0[:,0] = np.random.randn(n) * 0.5 + 0.4
        class_0[:,1] = np.random.randn(n) * 0.5 + 0.4
        class_1[:,0] = np.random.randn(n) * 0.5 + -0.3
        class_1[:,1] = np.random.randn(n) * 0.5 + -0.5

        class_0 = np.clip(class_0, -1, 1)
        class_1 = np.clip(class_1, -1, 1)
    elif level == 3:
        # Generate data for level 3
        class_0[:,0] = np.random.randn(n) * 0.3 + 0.1
        class_0[:,1] = np.random.randn(n) * 0.3 + 0.2
        class_1_r = np.random.randn(n) * 0.3 + 0.8
        class_1_theta = np.random.uniform(0, 2*np.pi, size = (n))
        class_1[:,0] = class_1_r * np.cos(class_1_theta)
        class_1[:,1] = class_1_r * np.sin(class_1_theta)

        class_0 = np.clip(class_0, -1, 1)
        class_1 = np.clip(class_1, -1, 1)

    elif level == 4:
        class_0[0:n//2,0] = np.random.randn(n//2) * 0.3 + 0.5
        class_0[0:n//2,1] = np.random.randn(n//2) * 0.3 + 0.5
        class_0[n//2:,0] = np.random.randn(n//2) * 0.3 + -0.5
        class_0[n//2:,1] = np.random.randn(n//2) * 0.3 + -0.5

        class_1[0:n//2,0] = np.random.randn(n//2) * 0.3 + 0.5
        class_1[0:n//2,1] = np.random.randn(n//2) * 0.3 + -0.5
        class_1[n//2:,0] = np.random.randn(n//2) * 0.3 + -0.5
        class_1[n//2:,1] = np.random.randn(n//2) * 0.3 + 0.5

        class_0 = np.clip(class_0, -1, 1)
        class_1 = np.clip(class_1, -1, 1)

    
    return class_0, class_1

def plot_data(class_0, class_1):
    # Plot data
    plt.scatter(class_0[:,0], class_0[:,1], c = 'r', marker = 'o', label = 'Class 0')
    plt.scatter(class_1[:,0], class_1[:,1], c = 'b', marker = 'x', label = 'Class 1')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def generate(levels, plot = False):
    for level in levels:
        class_0, class_1 = generate_data(level)

        if plot:
            plot_data(class_0, class_1)
    
        # write to json file
        data = {}
        
        data['class_0'] = class_0.tolist()
        data['class_1'] = class_1.tolist()

        with open('data/input_data_level_' + str(level) + '.json', 'w') as outfile:
            json.dump(data, outfile)

