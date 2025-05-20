import lightning as L
import torch
from torch import nn
# Assuming 'utils' and 'CellType' are defined elsewhere and imported correctly
from utils import CellType


class Decoder(L.LightningModule):
    def __init__(
        self,
        cell: str,
        hidden_size: int,
        layers: int,
        bidirectional: int, # Flag indicating if the encoder was bidirectional (0 for False, 1 for True)
        input_embedding_size: int,
        dropout: float,
    ):
        """
        Decoder module for a sequence-to-sequence model.

        Args:
            cell (str): Type of RNN cell to use (e.g., "LSTM", "GRU", "RNN").
                        Expected to be a key in `CellType` enum.
            hidden_size (int): The number of features in the hidden state of the RNN.
            layers (int): The number of recurrent layers.
            bidirectional (int): Indicates if the encoder was bidirectional (0 or 1).
                                 This influences the decoder's input embedding dimension.
            input_embedding_size (int): The base size of the input embedding for each token.
            dropout (float): Dropout probability for the RNN layers.
        """
        super(Decoder, self).__init__()
        self.cell = CellType[cell].value # Get the actual RNN cell class (e.g., nn.LSTM)
        self.hidden_size = hidden_size
        self.layers = layers
        self.input_embedding_size = input_embedding_size
        # Adjusts for encoder's bidirectionality: 1 if encoder unidirectional, 2 if bidirectional.
        self.bidirectional = bidirectional + 1

        self.vocab = [ # Target vocabulary (Hindi script characters and special tokens)
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
            "<PAD>",    # Padding token
            "<START>",  # Start of sequence token
            "<STOP>",   # End of sequence token
        ]
        # Embedding layer for target vocabulary tokens.
        # Dimension is scaled if the encoder was bidirectional.
        self.embedding = nn.Embedding(
            len(self.vocab),
            self.input_embedding_size * self.bidirectional,
        )
        # Decoder RNN is typically unidirectional.
        self.rnn = self.cell(
            self.input_embedding_size * self.bidirectional, # Input size matches embedding output
            self.hidden_size,
            self.layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False, # Decoder RNN is unidirectional
        )
        # Fully connected layer to map RNN output to vocabulary logits.
        self.fc = nn.Linear(
            in_features=self.hidden_size,
            out_features=len(self.vocab),
        )

    def normal_forward(self, hiddens):
        """
        Generates sequences step-by-step using the model's own predictions (greedy decoding).

        Args:
            hiddens: The initial hidden state from the encoder.
                     For LSTM, tuple (h_n, c_n); for GRU/RNN, h_n.
        """
        # Determine batch_size from the structure of the hidden state
        batch_size = len(hiddens[0])
        if isinstance(hiddens, tuple): # For LSTMs, hiddens is (h_0, c_0)
            batch_size = len(hiddens[0][0])

        # Initialize decoder input with <START> token for each sequence in batch
        seqs = torch.IntTensor([self.vocab.index("<START>")] * batch_size).to(
            device="cuda" # TODO: Consider using self.device for flexibility
        )

        output = torch.IntTensor([]).to(device="cuda") # Stores the output logits sequence
        # Generate sequence up to a maximum length (e.g., 50 tokens)
        for i in range(50):
            embeddings = self.embedding(seqs) # Embed the current input token(s)
            embeddings = embeddings.unsqueeze(1) # Add sequence dimension: (batch, 1, embed_dim)

            rnn_output, hiddens = self.rnn(embeddings, hiddens) # Pass through RNN
            logits = self.fc(rnn_output.squeeze(1)) # Get logits over vocabulary
            output = torch.cat((output, logits.unsqueeze(0)), 0) # Collect logits

            # Check for <STOP> token generation to terminate early
            all_seqs = torch.argmax(output, axis=-1) # Get predicted token indices for all steps so far
            # If all sequences in the batch have generated a <STOP> token, break
            if torch.all(torch.any(all_seqs == self.vocab.index("<STOP>"), axis=-1)):
                break

            # Use the predicted token (greedy) as the input for the next step
            seqs = torch.argmax(logits, axis=-1)

        return output.transpose(0, 1).contiguous() # (batch_size, seq_len, vocab_size)

    def teacher_forcing_forward(self, hiddens, tf_input_seqs):
        """
        Generates sequences using teacher forcing (feeding true target tokens as input).

        Args:
            hiddens: The initial hidden state from the encoder.
            tf_input_seqs (torch.Tensor): Ground truth target sequences.
                                         Shape: (batch_size, target_seq_len).
        """
        # Prepare input for teacher forcing by prepending <START> token
        start_seq = (
            torch.IntTensor([self.vocab.index("<START>")] * tf_input_seqs.shape[0])
            .to(device="cuda") # TODO: Consider using self.device
            .unsqueeze(1) # Shape: (batch_size, 1)
        )

        # Concatenate <START> token with the ground truth target sequences
        tf_input_seqs = torch.cat((start_seq, tf_input_seqs), axis=1) # Shape: (batch_size, target_seq_len + 1)
        tf_embeddings = self.embedding(tf_input_seqs) # Embed the entire sequence

        # Initialize output tensor to store logits for each token
        output = torch.zeros(
            len(tf_input_seqs[0]), tf_input_seqs.shape[0], len(self.vocab) # (seq_len+1, batch, vocab_size)
        ).to(device="cuda")

        # Process the sequence token by token using the embedded ground truth tokens
        for i in range(len(tf_embeddings[0])): # Iterate through the sequence length
            embeddings = tf_embeddings[:, i] # Current token's embeddings for all in batch
            embeddings = embeddings.unsqueeze(1) # Add sequence dimension: (batch, 1, embed_dim)

            rnn_output, hiddens = self.rnn(embeddings, hiddens) # Pass through RNN
            logits = self.fc(rnn_output.squeeze(1)) # Get logits

            output[i] += logits # Store logits for the current time step

        return output.transpose(0, 1).contiguous() # (batch_size, seq_len+1, vocab_size)

    def forward(self, hiddens, tf_input_seqs=None):
        """
        Main forward pass for the decoder. Switches between teacher forcing and normal decoding.

        Args:
            hiddens: The initial hidden state from the encoder.
            tf_input_seqs (torch.Tensor, optional): Target sequences for teacher forcing.
                                                    If None, normal (e.g., greedy) decoding is used.
                                                    Defaults to None.
        """
        if tf_input_seqs is not None:
            # Use teacher forcing if ground truth target sequences are provided
            output = self.teacher_forcing_forward(hiddens, tf_input_seqs)
        else:
            # Otherwise, use normal (e.g., greedy) decoding
            output = self.normal_forward(hiddens)

        return output
