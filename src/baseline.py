import torch
import torch.nn as nn

class BaselineMLP(nn.Module):
    def __init__(self, input_dim=96, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10)
        )

    def forward(self, x):
        return self.net(x)
