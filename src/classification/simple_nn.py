import torch
import torch.nn as nn
import torch.nn.functional as F



class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_rate=0.5):
        super().__init__()
        self.model = nn.ModuleList([
            nn.Linear(input_size, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size1, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        
            nn.Linear(hidden_size2, output_size)
        ])
        
    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        
        return x




if __name__ == "__main__":
    pass
