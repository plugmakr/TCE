import torch
import torch.nn as nn

class SimpleTCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(96, 64)
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.out(x)
