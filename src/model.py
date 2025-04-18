import torch
import torch.nn.functional as F
import torch.nn as nn


# MLP Architecture
class MLP(nn.Module):
    def __init__(self, n_features, n_hidden_layers=4, hidden_size=128, p_dropout=0.0):
        super(MLP, self).__init__()

        self.p_dropout = p_dropout
        self.n_hidden_layers = n_hidden_layers

        self.hidden_layers = nn.ModuleList([
            nn.Linear(n_features, hidden_size)
        ])
        for _ in range(n_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))

        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.hidden_layers[0](x))
        x = F.dropout(x, p=self.p_dropout, training=self.training)

        for i in range(1, self.n_hidden_layers):
            x = F.relu(self.hidden_layers[i](x))
            x = F.dropout(x, p=self.p_dropout, training=self.training)

        x = self.output_layer(x)
        x = torch.sigmoid(x)

        return x


# Logit Architecture
class Logit(nn.Module):
    def __init__(self, n_features, p_dropout=0):
        super(Logit, self).__init__()

        self.p_dropout = 0  # not included for Logit model
        self.lin1 = nn.Linear(n_features, 1)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.sigmoid(x)

        return x
