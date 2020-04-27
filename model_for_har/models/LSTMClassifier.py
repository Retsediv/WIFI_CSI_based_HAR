import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x.double(), (h0.double(), c0.double()))
        out = self.fc(out[:, -1, :].double())

        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(1), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(1), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]

        return [t for t in (h0, c0)]
