import lightning as L
import torch
from torch import nn
# Assuming 'utils' and 'CellType' are defined elsewhere and imported correctly
from utils import CellType


class Encoder(L.LightningModule):
    def __init__(
        self,
        cell: str,
        hidden_size: int,
        layers: int,
        bidirectional: bool, # This boolean flag indicates if the RNN should be bidirectional
        input_embedding_size: int,
        dropout: float,
    ):
        """
        Encoder module for a sequence-to-sequence model.

        Args:
            cell (str): Type of RNN cell to use (e.g., "LSTM", "GRU", "RNN").
                        Must be a key in the `CellType` enum.
            hidden_size (int): The number of features in the hidden state of the RNN
                               (for each direction if bidirectional).
            layers (int): The number of recurrent layers.
            bidirectional (bool): If True, a bidirectional RNN is used. Otherwise, a unidirectional RNN.
            input_embedding_size (int): The dimensionality of the input token embeddings.
            dropout (float): Dropout probability for the RNN layers.
        """
        super(Encoder, self).__init__()
        self.cell = CellType[cell].value # Retrieves the RNN cell class (e.g., nn.LSTM, nn.GRU) from the CellType enum
        self.hidden_size = hidden_size
        self.layers = layers
        self.input_embedding_size = input_embedding_size
        # self.bidirectional is used as a factor: 1 for unidirectional, 2 for bidirectional.
        # If input `bidirectional` is True (1), self.bidirectional becomes 2.
        # If input `bidirectional` is False (0), self.bidirectional becomes 1.
        self.bidirectional = bidirectional + 1

        self.vocab = [ # Defines the vocabulary for the encoder's input
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
            "<PAD>", # Padding token
        ]
        self.embedding = nn.Embedding(len(self.vocab), self.input_embedding_size) # Embedding layer
        self.rnn = self.cell(
            self.input_embedding_size, # Input size to RNN
            self.hidden_size,          # Hidden size for each RNN layer/direction
            self.layers,               # Number of RNN layers
            batch_first=True,          # Input/output tensors will have batch size as the first dimension
            dropout=dropout,           # Dropout rate for RNN layers (except the last one)
            bidirectional=bidirectional, # Use the original boolean flag for the nn.RNN module
        )
        # Fully connected layer to process the RNN hidden states.
        # If bidirectional, it halves the feature dimension, presumably to combine fwd/bwd states later.
        # `in_features` is `self.hidden_size` (size of one direction's hidden state).
        # `out_features` is `self.hidden_size // self.bidirectional` (i.e., //1 or //2).
        self.fc = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size // self.bidirectional, # Results in hidden_size or hidden_size/2
        )

    def concat_out_tensors(self, tensors: torch.Tensor):
        """
        Concatenates the hidden states from forward and backward RNNs for each layer.
        This is used when the RNN is bidirectional. The input `tensors` are assumed
        to be the output of the `self.fc` layer, having shape
        (num_layers * 2, batch_size, hidden_size / 2).

        Args:
            tensors (torch.Tensor): A tensor containing stacked hidden states
                                    (e.g., [layer1_fwd, layer1_bwd, layer2_fwd, layer2_bwd,...])
                                    after processing by the FC layer. Each sub-tensor has
                                    dimension `hidden_size / 2`.

        Returns:
            torch.Tensor: A tensor where forward and backward hidden states for each layer
                          are concatenated. Shape: (num_layers, batch_size, hidden_size).
        """
        # Initializes an empty integer tensor on CUDA. This will be filled with concatenated states.
        output = torch.IntTensor([]).to(device="cuda")

        # Iterates over the layers, taking pairs of tensors (forward and backward states for a layer)
        # `self.layers * 2` because `tensors` contains forward and backward states separately for each layer.
        for i in range(0, self.layers * 2, 2):
            # Concatenates the i-th (forward) and (i+1)-th (backward) hidden states along the last dimension.
            # `tensors[i]` and `tensors[i+1]` are (batch_size, hidden_size / 2).
            # `concat` becomes (batch_size, hidden_size).
            # `unsqueeze(0)` adds a new dimension at the beginning: (1, batch_size, hidden_size).
            concat = torch.cat((tensors[i], tensors[i + 1]), axis=-1).unsqueeze(0)
            # Appends the concatenated state for the current layer to the output tensor.
            output = torch.cat((output, concat), 0)

        return output

    def forward(self, input_seqs: torch.Tensor):
        """
        Defines the forward pass of the encoder.

        Args:
            input_seqs (torch.Tensor): A batch of input sequences (indices).
                                       Shape: (batch_size, sequence_length).

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                The final hidden state(s) of the encoder.
                If LSTM: a tuple (h_n, c_n), where h_n and c_n are the final hidden and cell states.
                         Shape of each: (num_layers, batch_size, hidden_size) if bidirectional,
                                        (num_layers, batch_size, hidden_size) if unidirectional (after fc).
                If GRU/RNN: the final hidden state h_n.
                            Shape: (num_layers, batch_size, hidden_size) if bidirectional,
                                   (num_layers, batch_size, hidden_size) if unidirectional (after fc).
        """
        embeddings = self.embedding(input_seqs) # Shape: (batch_size, seq_len, input_embedding_size)

        # `output` from RNN: all hidden states from the last layer.
        #   Shape: (batch_size, seq_len, hidden_size * self.bidirectional_factor)
        # `hidden_state` from RNN (h_n, or (h_n, c_n) for LSTM):
        #   Shape of h_n/c_n: (num_layers * self.bidirectional_factor, batch_size, hidden_size)
        output, hidden_state = self.rnn(embeddings)

        # `self.bidirectional` here is the factor (1 or 2)
        if isinstance(hidden_state, tuple): # Checks if the RNN is LSTM (returns tuple (h_n, c_n))
            # hidden_state[0] is h_n, hidden_state[1] is c_n
            # Pass h_n and c_n through the fully connected layer.
            # Input to fc: (num_layers * factor, batch, hidden_size)
            # Output of fc: (num_layers * factor, batch, hidden_size // factor)
            processed_h = self.fc(hidden_state[0])
            processed_c = self.fc(hidden_state[1])

            if self.bidirectional == 2: # If the RNN was bidirectional
                # Concatenate the forward and backward parts of h_n and c_n
                return (
                    self.concat_out_tensors(processed_h), # Shape: (num_layers, batch, hidden_size)
                    self.concat_out_tensors(processed_c), # Shape: (num_layers, batch, hidden_size)
                )
            else: # Unidirectional RNN
                # FC output for unidirectional is (num_layers, batch, hidden_size)
                return processed_h, processed_c
        else: # For GRU or basic RNN, hidden_state is just h_n
            processed_h = self.fc(hidden_state) # Apply FC layer

            if self.bidirectional == 2: # If the RNN was bidirectional
                # Concatenate the forward and backward parts of h_n
                return self.concat_out_tensors(processed_h) # Shape: (num_layers, batch, hidden_size)
            else: # Unidirectional RNN
                # FC output for unidirectional is (num_layers, batch, hidden_size)
                return processed_h
