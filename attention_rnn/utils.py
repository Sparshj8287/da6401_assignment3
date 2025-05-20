from enum import Enum

from torch import nn


class CellType(Enum):
    rnn = nn.RNN
    lstm = nn.LSTM
    gru = nn.GRU
