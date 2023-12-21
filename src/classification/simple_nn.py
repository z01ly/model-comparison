import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, out_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_size, out_size)
        self.short_cut = nn.Linear(in_size, out_size)

    def forward(self, x):
        # residual = x
        residual = self.short_cut(x)
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out += residual 
        out = self.relu(out)

        return out



class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_rate=0.5):
        super().__init__()
        self.model = nn.ModuleList([
            # nn.Linear(input_size, 256),
            # nn.BatchNorm1d(hidden_size1),
            # nn.ReLU(),
            # nn.Dropout(dropout_rate),
            ResidualBlock(input_size, 256),
            
            # nn.Linear(256, 128),
            # nn.BatchNorm1d(hidden_size2),
            # nn.ReLU(),
            # nn.Dropout(dropout_rate),
            ResidualBlock(256, 128),

            # nn.Linear(128, 64),
            # nn.ReLU(),
            ResidualBlock(128, 64),

            # nn.Linear(64, 32),
            # nn.ReLU(),
            ResidualBlock(64, 32),
        
            nn.Linear(32, output_size)
        ])
        
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        
        return x
