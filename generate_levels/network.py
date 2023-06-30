import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import json

class Network(torch.nn.Module):
    def __init__(self, lr = 0.1):
        super(Network, self).__init__()
        self.learning_rate = lr
        self.input_layer = torch.nn.Linear(4, 4)
        self.hidden_layer_1 = torch.nn.Linear(4, 4)
        self.hidden_layer_2 = torch.nn.Linear(4, 2)
        self.output_layer = torch.nn.Linear(2, 1)
        self.activation = torch.nn.ReLU()
        self.activation_final = torch.nn.Sigmoid()
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr = self.learning_rate)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hidden_layer_1(x)
        x = self.activation(x)
        x = self.hidden_layer_2(x)
        x = self.activation(x)
        x = self.output_layer(x)
        x = self.activation_final(x)
        return x
    
    def to_json(self, path):
        # Get weights
        weights = {}
        weights['input_layer'] = self.input_layer.weight.detach().numpy().tolist()
        weights['hidden_layer_1'] = self.hidden_layer_1.weight.detach().numpy().tolist()
        weights['hidden_layer_2'] = self.hidden_layer_2.weight.detach().numpy().tolist()
        weights['output_layer'] = self.output_layer.weight.detach().numpy().tolist()

        biases = {}
        biases['input_layer'] = self.input_layer.bias.detach().numpy().tolist()
        biases['hidden_layer_1'] = self.hidden_layer_1.bias.detach().numpy().tolist()
        biases['hidden_layer_2'] = self.hidden_layer_2.bias.detach().numpy().tolist()
        biases['output_layer'] = self.output_layer.bias.detach().numpy().tolist()

        data = {}
        data['weights'] = weights
        data['biases'] = biases

        # Save weights
        with open(path, 'w') as json_file:
            json.dump(data, json_file)

def train(net, data):
    # Get data
    x_0 = torch.tensor(data['class_0'], dtype = torch.float32)
    x_1 = torch.tensor(data['class_1'], dtype = torch.float32)

    # Concatenate data
    x = torch.cat((x_0, x_1), 0)

    # Create feature maps
    x_feature_map = x ** 2

    # Concatenate feature maps
    x = torch.cat((x, x_feature_map), 1)

    # Create labels
    y = torch.tensor([[0.0] for i in range(len(x_0))] + [[1.0] for i in range(len(x_1))], dtype = torch.float32)

    # Create training loop
    for epoch in tqdm.tqdm(range(10000)):
        # Forward pass
        y_pred = net(x)

        # Calculate loss
        loss = net.loss(y_pred, y)
        # print(epoch, loss.item())

        # Zero gradients, backward pass, update weights
        net.optimizer.zero_grad()
        loss.backward()
        net.optimizer.step()

        # Update tqdm
        if (epoch % 5000 == 0):
            tqdm.tqdm.write('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.item()))

            # Show model outputs
            y_pred = y_pred.detach().numpy()
            plt.scatter(x[:, 0], x[:, 1], c = y_pred)
            plt.colorbar()
            plt.show()


    
    # Return model
    return net


