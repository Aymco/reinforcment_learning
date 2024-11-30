import torch
import torch.nn as nn
import numpy as np

class torchNN(nn.Module):
    def __init__(self, layers=[8, 4, 3, 2]):
        super(torchNN, self).__init__()
        self.layers = layers

        self.Layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.Layers.append(nn.Linear(layers[i], layers[i + 1]))

        self.mutation_rate = 0.5
        self.mutation_chance = 0.4

    def forward_prop(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = x.view(1, -1)
        print(x)
        for i in range(len(self.Layers) - 1):
            x = torch.relu(self.Layers[i](x))
        x = self.Layers[-1](x)
        print(x)
        return x[:, 0], x[:, 1]

    def mutate(self, bestnn):
        for i, layer in enumerate(self.Layers):
            with torch.no_grad():
                weight_variation = self.__get_random_variation(layer.weight.size())
                layer.weight += bestnn.Layers[i].weight * weight_variation
                bias_variation = self.__get_random_variation(layer.bias.size())
                layer.bias += bestnn.Layers[i].bias * bias_variation

                layer.weight.clamp_(-1, 1)
                layer.bias.clamp_(-1, 1)

    def __get_random_variation(self, size):
        mask = torch.rand(size) < self.mutation_chance
        return self.mutation_rate * (torch.rand(size) * 2 - 1) * mask

    def save(self, path):
        torch.save(self.state_dict(), path)
        print('Model saved !')

    def load(self, path):
        self.load_state_dict(torch.load(path))
        print('Model loaded ! :', path)

    def copy_weights(self, other):
        for i, layer in enumerate(self.Layers):
            with torch.no_grad():
                layer.weight.copy_(other.Layers[i].weight)
                layer.bias.copy_(other.Layers[i].bias)
    
    def init(self):
        for layer in self.Layers:
            with torch.no_grad():
                layer.weight = nn.Parameter(torch.rand(layer.weight.size()) - 0.5)
                layer.bias = nn.Parameter(torch.rand(layer.bias.size()) - 0.5)
        return self
