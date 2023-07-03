'''
Author: Felix O'Mahony
Date: 29/06/2023
License: See license.txt file
-----------
Description: this file generates data for the game.
The network architecture is a simple 3-layer network with 4 neurons in the first hidden layer and 2 neurons in the second hidden layer.
The input layer has 4 neurons, the 2 input ordinates and a feature map of each which is the square of the ordinates.
The output layer has 1 neuron, the data class.'''

import numpy as np
import torch
import json

import network as network
import generate_data as generate_data


levels = [0,1,2,3,4]

if __name__ == '__main__':
    # Generate data
    generate_data.generate(levels, True)

    for level in levels:
        # Load data
        with open('data/input_data_level_'+str(level)+'.json') as json_file:
            data = json.load(json_file)
        
        # For each level
        # Create network
        net = network.Network()
        net = network.train(net, data)
        # Write model to json file
        net.to_json('data/model_level_'+str(level)+'.json')

