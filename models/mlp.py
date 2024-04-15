import numpy as np
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, args, nf=400):
        super(MLP, self).__init__()
        self.num_classes = args.n_classes
        self.input_size = np.prod(args.input_size)
        self.hidden = nn.Sequential(nn.Linear(self.input_size, nf),
                                    nn.ReLU(True),
                                    nn.Linear(nf, nf),
                                    nn.ReLU(True))

        self.linear = nn.Linear(nf, self.num_classes)

    def return_hidden(self,x):
        x = x.view(-1, self.input_size)
        return self.hidden(x)

    def forward(self, x):
        out = self.return_hidden(x)
        return self.linear(out)