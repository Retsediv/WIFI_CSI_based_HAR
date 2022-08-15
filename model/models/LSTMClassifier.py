import torch
from torch import nn
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LSTMClassifier(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_layers, dropout, bidirectional, num_classes, batch_size):
        super(LSTMClassifier, self).__init__()
        self.arch = 'lstm'
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_dir = 2 if bidirectional else 1
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.hidden2label = nn.Sequential(
            nn.Linear(hidden_dim * self.num_dir, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.hidden = None

    def init_hidden(self, batch_size):
        h0 = nn.Parameter(nn.init.xavier_uniform_(
                torch.Tensor(self.num_dir * self.num_layers, batch_size, self.hidden_dim).type(torch.DoubleTensor)
            ), requires_grad=True).to(device)

        c0 = nn.Parameter(nn.init.xavier_uniform_(
                torch.Tensor(self.num_dir * self.num_layers, batch_size, self.hidden_dim).type(torch.DoubleTensor)
            ), requires_grad=True).to(device)

        return h0, c0

    def forward(self, x):  # x is (batch_size, sequence_size, num_of_features)
        if self.hidden is None:
            self.hidden = self.init_hidden(x.size(0))

        # See: https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426/2
        lstm_out, _ = self.lstm(x, self.hidden)

        y = self.hidden2label(lstm_out[:, -1, :])

        return y
