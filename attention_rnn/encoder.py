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
        bidirectional: bool, # Boolean flag: True for bidirectional, False for unidirectional
        input_embedding_size: int,
        dropout: float,
    ):
        """
        Encoder module for a sequence-to-sequence model, potentially for use with attention.

        Args:
            cell (str): Type of RNN cell (e.g., "LSTM", "GRU").
            hidden_size (int): Size of the RNN hidden state (per direction).
            layers (int): Number of RNN layers.
            bidirectional (bool): Specifies if the RNN is bidirectional.
            input_embedding_size (int): Dimensionality of token embeddings.
            dropout (float): Dropout rate for RNN layers.
        """
        super(Encoder, self).__init__()
        self.cell = CellType[cell].value # RNN cell class
        self.hidden_size = hidden_size
        self.layers = layers
        self.input_embedding_size = input_embedding_size
        # self.bidirectional factor: 1 for unidirectional, 2 for bidirectional
        self.bidirectional = bidirectional + 1

        self.vocab = [ # Source vocabulary
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
            "<PAD>",
        ]
        self.embedding = nn.Embedding(len(self.vocab), self.input_embedding_size)
        self.rnn = self.cell(
            self.input_embedding_size,
            self.hidden_size,
            self.layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional, # Pass the original boolean flag to nn.RNN module
        )
        # FC layer to potentially transform hidden states, halving features if bidirectional
        # This is applied to the final hidden state (h_n, c_n) before concatenation.
        self.fc = nn.Linear(
            in_features=self.hidden_size, # Input is hidden_size (from one direction)
            out_features=self.hidden_size // self.bidirectional, # Output is hidden_size or hidden_size/2
        )

    def concat_out_tensors(self, tensors: torch.Tensor):
        """
        Concatenates hidden states from bidirectional RNN layers after FC transformation.
        Input `tensors` shape: (num_layers * 2, batch_size, hidden_size / 2).
        Output shape: (num_layers, batch_size, hidden_size).

        Args:
            tensors (torch.Tensor): Stacked hidden states (e.g., [L1_fwd, L1_bwd, L2_fwd, ...]).
        """
        # Initializes an empty integer tensor on CUDA. This will be filled with concatenated states.
        output = torch.IntTensor([]).to(device="cuda") # TODO: Consider using tensors.device

        # Iterates over the layers, taking pairs of tensors (forward and backward states for a layer)
        for i in range(0, self.layers * 2, 2):
            # Concatenates the i-th (forward) and (i+1)-th (backward) hidden states along the last dimension.
            concat = torch.cat((tensors[i], tensors[i + 1]), axis=-1).unsqueeze(0)
            output = torch.cat((output, concat), 0)

        return output

    def forward(self, input_seqs: torch.Tensor):
        """
        Forward pass of the encoder. Returns all encoder outputs and the final hidden state.

        Args:
            input_seqs (torch.Tensor): Batch of input sequences (token indices).
                                       Shape: (batch_size, sequence_length).

        Returns:
            Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
                - output (torch.Tensor): All hidden states from the last RNN layer.
                  Shape: (batch_size, seq_len, hidden_size * bidir_factor).
                - hidden_state: Final hidden state(s).
                  If LSTM: tuple (h_n, c_n). If GRU/RNN: h_n.
                  Processed by FC layer and concatenated if bidirectional.
                  Shape of h_n/c_n after processing: (num_layers, batch_size, hidden_size).
        """
        embeddings = self.embedding(input_seqs) # (batch_size, seq_len, input_embedding_size)

        # `output`: hidden states from all time steps of the last layer
        # `hidden_state`: final hidden state (h_n or (h_n, c_n)) from all layers
        output, hidden_state = self.rnn(embeddings)

        if isinstance(hidden_state, tuple): # LSTM case: hidden_state is (h_n, c_n)
            # Apply FC layer to h_n and c_n separately
            processed_h = self.fc(hidden_state[0])
            processed_c = self.fc(hidden_state[1])
            
            final_hidden_state_tuple = (processed_h, processed_c)

            if self.bidirectional == 2: # If bidirectional
                # Concatenate forward and backward states for h_n and c_n
                final_hidden_state_tuple = (
                    self.concat_out_tensors(processed_h),
                    self.concat_out_tensors(processed_c),
                )
            # Return all encoder outputs and the final (processed) hidden state tuple
            return output, final_hidden_state_tuple
        else: # GRU or RNN case: hidden_state is h_n
            processed_h = self.fc(hidden_state) # Apply FC layer
            
            final_hidden_state = processed_h
            if self.bidirectional == 2: # If bidirectional
                # Concatenate forward and backward states
                final_hidden_state = self.concat_out_tensors(processed_h)
            # Return all encoder outputs and the final (processed) hidden state
            return output, final_hidden_state
