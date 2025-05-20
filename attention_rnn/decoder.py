import lightning as L
import torch
from torch import nn
from utils import CellType


class Decoder(L.LightningModule):
    def __init__(
        self,
        cell: str,
        hidden_size: int,
        layers: int,
        bidirectional: int,
        input_embedding_size: int,
        dropout: float,
    ):
        super(Decoder, self).__init__()
        self.cell = CellType[cell].value
        self.hidden_size = hidden_size
        self.layers = layers
        self.input_embedding_size = input_embedding_size
        self.bidirectional = bidirectional + 1

        self.vocab = [
    "ँ",
    "ं",
    "ः",
    "अ",
    "आ",
    "इ",
    "ई",
    "उ",
    "ऊ",
    "ऋ",
    "ए",
    "ऐ",
    "ऑ",
    "ओ",
    "औ",
    "क",
    "ख",
    "ग",
    "घ",
    "ङ",
    "च",
    "छ",
    "ज",
    "झ",
    "ञ",
    "ट",
    "ठ",
    "ड",
    "ढ",
    "ण",
    "त",
    "थ",
    "द",
    "ध",
    "न",
    "प",
    "फ",
    "ब",
    "भ",
    "म",
    "य",
    "र",
    "ल",
    "व",
    "श",
    "ष",
    "स",
    "ह",
    "़",
    "ा",
    "ि",
    "ी",
    "ु",
    "ू",
    "ृ",
    "ॅ",
    "े",
    "ै",
    "ॉ",
    "ो",
    "ौ",
    "्",
    "ॐ",
            "<PAD>",
            "<START>",
            "<STOP>",
        ]
        self.embedding = nn.Embedding(
            len(self.vocab),
            self.input_embedding_size * self.bidirectional,
        )
        self.rnn = self.cell(
            (self.input_embedding_size + self.hidden_size) * self.bidirectional,
            self.hidden_size,
            self.layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.fc = nn.Linear(
            in_features=self.hidden_size,
            out_features=len(self.vocab),
        )
        self.w_att_fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.u_att_fc = nn.Linear(
            self.hidden_size * self.bidirectional, self.hidden_size
        )
        self.v_att_fc = nn.Linear(
            self.hidden_size, self.hidden_size * self.bidirectional
        )
        self.activation = nn.Tanh()
        self.att_activation = nn.Softmax(dim=0)

    def bahdanau_attention(self, output, hidden, embeddings):
        if isinstance(hidden, tuple):
            hidden = hidden[0][0]
        else:
            hidden = hidden[0]

        ws = self.w_att_fc(hidden)
        uh = self.u_att_fc(output).transpose(0, 1).contiguous()
        alpha = self.att_activation(self.v_att_fc(self.activation(ws + uh)))
        c = torch.sum(alpha * output.transpose(0, 1).contiguous(), axis=0)
        x = torch.cat([c, embeddings], 1)

        return x

    def normal_forward(self, enc_output, hiddens):
        batch_size = len(hiddens[0])
        if isinstance(hiddens, tuple):
            batch_size = len(hiddens[0][0])

        seqs = torch.IntTensor([self.vocab.index("<START>")] * batch_size).to(
            device="cuda"
        )

        output = torch.IntTensor([]).to(device="cuda")
        for i in range(50):
            embeddings = self.embedding(seqs)
            embeddings = self.bahdanau_attention(enc_output, hiddens, embeddings)
            embeddings = embeddings.unsqueeze(1)

            rnn_output, hiddens = self.rnn(embeddings, hiddens)
            logits = self.activation(self.fc(rnn_output.squeeze(1)))
            output = torch.cat((output, logits.unsqueeze(0)), 0)

            all_seqs = torch.argmax(output, axis=-1)
            if torch.all(torch.any(all_seqs == self.vocab.index("<STOP>"), axis=-1)):
                break

            seqs = torch.argmax(logits, axis=-1)

        return output.transpose(0, 1).contiguous()

    def teacher_forcing_forward(self, enc_output, hiddens, tf_input_seqs):
        start_seq = (
            torch.IntTensor([self.vocab.index("<START>")] * tf_input_seqs.shape[0])
            .to(device="cuda")
            .unsqueeze(1)
        )

        tf_input_seqs = torch.cat((start_seq, tf_input_seqs), axis=1)
        tf_embeddings = self.embedding(tf_input_seqs)

        output = torch.zeros(
            len(tf_input_seqs[0]), tf_input_seqs.shape[0], len(self.vocab)
        ).to(device="cuda")

        for i in range(len(tf_embeddings[0])):
            embeddings = tf_embeddings[:, i]
            embeddings = self.bahdanau_attention(enc_output, hiddens, embeddings)
            embeddings = embeddings.unsqueeze(1)

            rnn_output, hiddens = self.rnn(embeddings, hiddens)
            logits = self.fc(rnn_output.squeeze(1))

            output[i] += logits

        return output.transpose(0, 1).contiguous()

    def forward(self, enc_output, hiddens, tf_input_seqs=None):
        if tf_input_seqs is not None:
            output = self.teacher_forcing_forward(enc_output, hiddens, tf_input_seqs)
        else:
            output = self.normal_forward(enc_output, hiddens)

        return output
