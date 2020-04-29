import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    def __init__(self, input_dim, hidden_dim, num_layers_lstm, out_classes_num, dropout=0.0, bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = num_layers_lstm

        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=num_layers_lstm,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=bidirectional)

        self.hidden2label = nn.Linear(hidden_dim, out_classes_num)

        self.hidden = None

    def forward(self, x):
        if self.hidden is None:
            self.hidden = self.init_hidden(x.size(0))

        out, _ = self.lstm(x, self.hidden)

        # print("out: ", out.shape)
        # print("out[:, -1, :]: ", out[:, -1, :].shape)
        #
        # print("self.hidden2label(out): ", self.hidden2label(out).shape)
        # print("self.hidden2label(out[:, -1, :]): ", self.hidden2label(out[:, -1, :]).shape)

        out = self.hidden2label(out)
        return out

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).double().to(device)
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).double().to(device)

        return h0, c0
