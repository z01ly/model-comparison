import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_features, in_features)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x += residual  # Add the residual connection
        x = self.relu(x)

        return x



class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_rate=0.5):
        super().__init__()
        self.model = nn.ModuleList([
            nn.Linear(input_size, 256),
            # nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            # nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            # nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),
        
            nn.Linear(32, output_size)
        ])
        
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        
        return x
