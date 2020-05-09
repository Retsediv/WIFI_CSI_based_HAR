import torch
from torch import nn

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class SimpleLSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    def __init__(self, input_dim, hidden_dim, num_layers_lstm, out_classes_num, dropout=0.0, bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = num_layers_lstm
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=num_layers_lstm,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=bidirectional,
                            bias=True)

        self.hidden2label = nn.Linear((hidden_dim * 2) if self.bidirectional else hidden_dim, out_classes_num)
        self.hidden = None

    def forward(self, x):
        if self.hidden is None:
            self.hidden = self.init_hidden(x.size(0))

        out, _ = self.lstm(x, self.hidden)
        out = self.hidden2label(out[:, -1, :])

        return out

    def init_hidden(self, batch_size):
        h0 = nn.Parameter(nn.init.xavier_uniform_(
                torch.Tensor(self.layer_dim, batch_size, self.hidden_dim).type(torch.DoubleTensor)
            ), requires_grad=True).to(device)

        c0 = nn.Parameter(nn.init.xavier_uniform_(
                torch.Tensor(self.layer_dim, batch_size, self.hidden_dim).type(torch.DoubleTensor)
            ), requires_grad=True).to(device)

        return h0, c0