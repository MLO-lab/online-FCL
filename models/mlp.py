import numpy as np
import torch.nn as nn

# class MLP(nn.Module):
#     def __init__(self, args, nf=128):
#         super(MLP, self).__init__()
#         self.args = args
#         self.num_classes = args.n_classes
#         self.input_size = np.prod(args.input_size)
#         self.hidden = nn.Sequential(nn.Linear(self.input_size, nf),
#                                     nn.ReLU(True),
#                                     nn.Linear(nf, nf),
#                                     nn.ReLU(True))

#         self.linear = nn.Linear(nf, self.num_classes)

#     def return_hidden(self,x):
#         if self.args.dataset_name == 'mnist':
#             x = x.view(-1, self.input_size)
#             return self.hidden(x)
#         else:
#             return self.hidden(x)
    

#     def forward(self, x):
#         out = self.return_hidden(x)
#         return self.linear(out)
    

# class MLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(MLP, self).__init__()
#         self.hidden = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.output = nn.Linear(hidden_dim, output_dim)
    
#     def forward(self, x):
#         x = self.hidden(x)
#         x = self.relu(x)
#         x = self.output(x)
#         return x

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.layers.append(nn.ReLU())
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x