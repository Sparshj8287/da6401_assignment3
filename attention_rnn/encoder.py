import lightning as L
import torch
from torch import nn
from utils import CellType


class Encoder(L.LightningModule):
    def __init__(
        self,
        cell: str,
        hidden_size: int,
        layers: int,
        bidirectional: bool,
        input_embedding_size: int,
        dropout: float,
    ):
        super(Encoder, self).__init__()
        self.cell = CellType[cell].value
        self.hidden_size = hidden_size
        self.layers = layers
        self.input_embedding_size = input_embedding_size
        self.bidirectional = bidirectional + 1

        self.vocab = [
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
            "<PAD>",
        ]
        self.embedding = nn.Embedding(len(self.vocab), self.input_embedding_size)
        self.rnn = self.cell(
            self.input_embedding_size,
            self.hidden_size,
            self.layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size // self.bidirectional,
        )

    def concat_out_tensors(self, tensors):
        output = torch.IntTensor([]).to(device="cuda")

        for i in range(0, self.layers * 2, 2):
            concat = torch.cat((tensors[i], tensors[i + 1]), axis=-1).unsqueeze(0)
            output = torch.cat((output, concat), 0)

        return output

    def forward(self, input_seqs):
        embeddings = self.embedding(input_seqs)

        output, hidden_state = self.rnn(embeddings)

        if isinstance(hidden_state, tuple):
            hidden_state = (
                self.fc(hidden_state[0]),
                self.fc(hidden_state[1]),
            )
            if self.bidirectional == 2:
                return (
                    output,
                    (
                        self.concat_out_tensors(hidden_state[0]),
                        self.concat_out_tensors(hidden_state[1]),
                    ),
                )
            else:
                return output, hidden_state
        else:
            hidden_state = self.fc(hidden_state)
            if self.bidirectional == 2:
                return output, self.concat_out_tensors(hidden_state)
            else:
                return output, hidden_state
