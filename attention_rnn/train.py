import argparse
import random
import uuid
from typing import Tuple

import lightning as L
import numpy as np # Not explicitly used in the visible snippet, but often present
import torch
import torchmetrics
from dataset import DakshinaDataModule # For loading and preparing the Dakshina dataset
from decoder import Decoder # The decoder part of the sequence-to-sequence model (with attention)
from encoder import Encoder # The encoder part of the sequence-to-sequence model (outputs all hidden states)
from lightning.pytorch.callbacks import ModelCheckpoint # For saving model checkpoints
from lightning.pytorch.loggers import WandbLogger # For logging to Weights & Biases
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence # For padding sequences to the same length
from word_accuracy import WordAccuracy # Custom metric for word-level accuracy

import wandb # For Weights & Biases integration


class RNN(L.LightningModule):
    def __init__(
        self,
        n_layers: int,
        cell: str,
        hidden_size: int,
        input_embedding_size: int,
        lr: float,
        bidirectional: bool,
        enc_dropout: float,
        dec_dropout: float,
    ):
        """
        Initializes the RNN model with specified architecture and hyperparameters.

        Args:
            n_layers (int): Number of layers in both encoder and decoder RNNs.
            cell (str): Type of RNN cell to use (e.g., "LSTM", "GRU").
            hidden_size (int): Size of the hidden state in RNN cells.
            input_embedding_size (int): Dimensionality of input token embeddings.
            lr (float): Learning rate for the optimizer.
            bidirectional (bool): Whether the encoder RNN is bidirectional.
            enc_dropout (float): Dropout rate for the encoder RNN.
            dec_dropout (float): Dropout rate for the decoder RNN.
        """
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_embedding_size = input_embedding_size
        self.lr = lr
        # Encoder and Decoder are initialized here.
        # The encoder now returns all its output states for attention.
        self.encoder = Encoder(
            cell,
            hidden_size,
            n_layers,
            bidirectional,
            input_embedding_size,
            enc_dropout,
        )
        self.decoder = Decoder( # Decoder now likely uses attention with encoder outputs
            cell,
            hidden_size,
            n_layers,
            bidirectional, # Pass bidirectionality flag (0 or 1)
            input_embedding_size,
            dec_dropout,
        )
        # Loss function, ignoring padding tokens.
        self.loss = nn.CrossEntropyLoss(ignore_index=self.decoder.vocab.index("<PAD>"))
        self.tf_prob = 100 # Initial teacher forcing probability.
        # Metrics for accuracy.
        self.word_train_acc = WordAccuracy(
            ignore_index=self.decoder.vocab.index("<PAD>")
        )
        # Duplicate self.word_train_acc initialization, likely a typo, keeping as is.
        self.word_train_acc = WordAccuracy(
            ignore_index=self.decoder.vocab.index("<PAD>")
        )
        self.word_val_acc = WordAccuracy(ignore_index=self.decoder.vocab.index("<PAD>"))
        self.train_acc = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=len(self.decoder.vocab),
            ignore_index=self.decoder.vocab.index("<PAD>"),
        )
        self.val_acc = torchmetrics.classification.Accuracy(
            task="multiclass",
            num_classes=len(self.decoder.vocab),
            ignore_index=self.decoder.vocab.index("<PAD>"),
        )

    def forward(self, input_seqs, tf_input_seqs=None, tf_prob: float = 0):
        """
        Defines the forward pass of the RNN model, passing encoder outputs to the decoder.

        Args:
            input_seqs (torch.Tensor): Input sequences for the encoder.
            tf_input_seqs (torch.Tensor, optional): Target sequences for teacher forcing. Defaults to None.
            tf_prob (float, optional): Probability of using teacher forcing (0-100). Defaults to 0.
        """
        # Encoder returns all its hidden states (output) and the final hidden state (hiddens).
        output, hiddens = self.encoder(input_seqs)

        # Decoder receives encoder's full output sequence for attention, and the final hidden state.
        if tf_prob and random.randrange(1, 100) <= tf_prob: # Corrected upper bound for randrange
            y = self.decoder(output, hiddens, tf_input_seqs) # Decoder with teacher forcing
        else:
            y = self.decoder(output, hiddens) # Decoder without teacher forcing

        return y

    def add_softmax_padding(self, t: torch.Tensor, c: int):
        """
        Pads a tensor with softmax distributions representing <PAD> tokens to match sequence length.

        Args:
            t (torch.Tensor): The tensor to pad (predictions). Shape (batch_size, seq_len, vocab_size).
            c (int): The number of padding steps to add.
        """
        vocab_len = len(self.decoder.vocab)
        # Creates a one-hot like vector for <PAD> token.
        softmax = torch.zeros(vocab_len).to(device="cuda") # TODO: Consider t.device
        softmax[self.decoder.vocab.index("<PAD>")] = 1 # Uses the correct <PAD> index.
        # Repeats padding for batch and padding steps.
        rep_softmax = softmax.repeat(t.shape[0], 1).unsqueeze(0).repeat(c, 1, 1)

        return torch.cat((t.transpose(0, 1), rep_softmax)).transpose(0, 1).contiguous()

    def pre_process_text(self, x: Tuple[str], y: Tuple[str]):
        """
        Converts raw text sequences (source and target) into padded integer tensors.

        Args:
            x (Tuple[str]): A batch of source text sequences.
            y (Tuple[str]): A batch of target text sequences.
        """
        input_seqs = []
        tf_input_seqs = []
        # Converts characters to indices and adds <STOP> to target.
        for x1, y1 in zip(x, y):
            input_seqs.append(
                torch.IntTensor([self.encoder.vocab.index(i) for i in x1])
            )
            tf_input_seqs.append(
                torch.IntTensor(
                    [self.decoder.vocab.index(i) for i in y1]
                    + [self.decoder.vocab.index("<STOP>")]
                )
            )
        # Pads sequences to the same length using respective <PAD> token indices.
        padded_input_seqs = pad_sequence(
            input_seqs, batch_first=True, padding_value=self.encoder.vocab.index("<PAD>") # Correct padding value
        ).to(device="cuda") # TODO: Consider self.device
        padded_tf_input_seqs = pad_sequence(
            tf_input_seqs, batch_first=True, padding_value=self.decoder.vocab.index("<PAD>") # Correct padding value
        ).to(device="cuda") # TODO: Consider self.device

        return (padded_input_seqs, padded_tf_input_seqs)

    def training_step(self, batch, batch_idx: int):
        """
        Performs a single training step including forward pass, loss calculation, and logging.

        Args:
            batch: A batch of data (source sequences, target sequences).
            batch_idx (int): The index of the current batch.
        """
        x, y = batch
        input_seqs, tf_input_seqs = self.pre_process_text(x, y)
        # Forward pass with teacher forcing.
        y_hat = self(input_seqs, tf_input_seqs, self.tf_prob)
        # Adjust sequence lengths of predictions and targets.
        diff = y_hat.shape[1] - tf_input_seqs.shape[1] # Corrected comparison

        if diff < 0: # Predictions shorter
            y_hat = self.add_softmax_padding(y_hat, -diff)
        elif diff > 0: # Predictions longer, pad targets
            padding = torch.full((tf_input_seqs.shape[0], diff), # Correct shape for padding
                                 fill_value=self.decoder.vocab.index("<PAD>"),
                                 device="cuda") # TODO: Consider self.device
            tf_input_seqs = torch.cat( # Corrected concatenation
                (
                    tf_input_seqs,
                    padding,
                ),
                dim=1 # Concatenate along sequence dimension
            ).contiguous()

        tf_input_seqs = tf_input_seqs.to(torch.long)
        # Calculate loss and log metrics.
        loss = self.loss(y_hat.view(-1, y_hat.size(-1)), tf_input_seqs.view(-1))
        y_hat_seq = torch.argmax(y_hat, axis=-1)
        self.train_acc(y_hat_seq, tf_input_seqs)
        self.word_train_acc(y_hat_seq, tf_input_seqs)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=len(x))
        self.log("word_train_acc", self.word_train_acc, prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=len(x))
        self.log("train_loss", loss, logger=True, prog_bar=True, on_step=True, on_epoch=False, batch_size=len(x))
        return loss

    def validation_step(self, batch, batch_idx: int):
        """
        Performs a single validation step, similar to training but without teacher forcing.

        Args:
            batch: A batch of validation data.
            batch_idx (int): The index of the current batch.
        """
        x, y = batch
        input_seqs, tf_input_seqs = self.pre_process_text(x, y)
        # Forward pass without teacher forcing for validation.
        y_hat = self(input_seqs)
        # Adjust sequence lengths.
        diff = y_hat.shape[1] - tf_input_seqs.shape[1] # Corrected comparison
        if diff < 0:
            y_hat = self.add_softmax_padding(y_hat, -diff)
        elif diff > 0: # Predictions longer, pad targets
            padding = torch.full((tf_input_seqs.shape[0], diff), # Correct shape for padding
                                 fill_value=self.decoder.vocab.index("<PAD>"),
                                 device="cuda") # TODO: Consider self.device
            tf_input_seqs = torch.cat( # Corrected concatenation
                (
                    tf_input_seqs,
                    padding,
                ),
                dim=1 # Concatenate along sequence dimension
            ).contiguous()

        tf_input_seqs = tf_input_seqs.to(torch.long)
        # Calculate loss and log validation metrics.
        loss = self.loss(y_hat.view(-1, y_hat.size(-1)), tf_input_seqs.view(-1))
        y_hat_seq = torch.argmax(y_hat, axis=-1)
        self.val_acc(y_hat_seq, tf_input_seqs)
        self.word_val_acc(y_hat_seq, tf_input_seqs)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=len(x))
        self.log("word_val_acc", self.word_val_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(x))
        self.log("val_loss", loss, logger=True, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(x))

    def on_train_epoch_end(self) -> None:
        """
        Called at the end of each training epoch to decay teacher forcing probability.
        """
        # Decays teacher forcing probability linearly over epochs.
        self.tf_prob = 100 - (self.trainer.current_epoch +1) * 100 / self.trainer.max_epochs # current_epoch is 0-indexed
        self.tf_prob = max(0, self.tf_prob) # Ensure probability is not negative

    def configure_optimizers(self):
        """
        Configures the optimizer (Adam) for training.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def train(args):
    """
    Main training function: sets up logging, data, model, and trainer, then starts training.

    Args:
        args (argparse.Namespace): Command-line arguments and hyperparameters.
    """
    # Sets up Weights & Biases logging or a local run ID.
    if args.log_location == "wandb":
        wandb.init(entity=args.wandb_entity, project=args.wandb_project)
        wandb.run.name = f"cell_{args.cell}_bd_{args.bidirectional}_bs_{args.batch_size}_l_{args.layers}_e_{args.epochs}_hs_{args.hidden_size}_ies_{args.input_embedding_size}_id_{wandb.run.id}"
        model_name = wandb.run.name
        # Corrected WandbLogger initialization
        wandb_logger = WandbLogger(name=model_name, project=args.wandb_project, entity=args.wandb_entity, log_model=False)
    else:
        model_name = str(uuid.uuid4())
        wandb_logger = None

    print("Training with the following hyperparameters:")
    print(args)
    # Initializes data module, model, and checkpoint callback.
    dakshina = DakshinaDataModule(args.dataset_path, args.batch_size)
    model = RNN(
        args.layers,
        args.cell,
        args.hidden_size,
        args.input_embedding_size,
        args.learning_rate,
        bool(args.bidirectional), # Ensure boolean conversion
        args.encoder_dropout,
        args.decoder_dropout,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=model_name,
        save_top_k=1,
        verbose=True,
        monitor="val_acc", # Monitor validation accuracy
        mode="max",
    )
    # Initializes and starts the PyTorch Lightning Trainer.
    trainer = L.Trainer(
        max_epochs=args.epochs,
        logger=wandb_logger,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        # Example GPU usage: gpus=1 if torch.cuda.is_available() else 0
    )
    trainer.fit(model=model, datamodule=dakshina)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an RNN based transliteration model with optional attention"
    )
    parser.add_argument(
        "--wandb_entity",
        "-we",
        type=str,
        default="myname",
        help="Wandb Entity used to track experiments.",
    )
    parser.add_argument(
        "--wandb_project",
        "-wp",
        type=str,
        default="myprojectname",
        help="Project name in Weights & Biases.",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=20, help="Number of epochs to train."
    )
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size.") # Default changed as per new config
    parser.add_argument(
        "--learning_rate", "-lr", type=float, default=0.00249, help="Learning rate."
    )
    parser.add_argument(
        "--layers", "-l", type=int, default=3, help="Number of stacked RNN Layers"
    )
    parser.add_argument(
        "--cell",
        "-c",
        type=str,
        default="gru", 
        choices=[
            "rnn",
            "lstm",
            "gru",
        ],
        help="RNN Cell Type",
    )
    parser.add_argument(
        "--hidden_size",
        "-h_s",
        type=int,
        default=256,
        help="Hidden Layer size in RNN",
    )
    parser.add_argument(
        "--input_embedding_size",
        "-i_e_s",
        type=int,
        default=128,
        help="Input embedding size in RNN",
    )
    parser.add_argument(
        "--bidirectional",
        "-bd",
        type=int,
        default=1,
        choices=[0, 1],
        help="Enable bidirectional support for cells (0 for False, 1 for True)",
    )
    parser.add_argument(
        "--encoder_dropout",
        "-e_d",
        type=float,
        default=0.2925,
        help="Dropout for encoder layers",
    )
    parser.add_argument(
        "--decoder_dropout",
        "-d_d",
        type=float,
        default=0.2925,
        help="Dropout for decoder layers",
    )
    parser.add_argument(
        "--dataset_path",
        "-d_p",
        type=str,
        default="/projects/data/astteam/repos/da6401_sparsh/da6401_assignment3/data/dakshina_dataset_v1.0/hi/lexicons",
        help="Path to dataset",
    )
    parser.add_argument(
        "--log_location",
        "-g",
        type=str,
        default="wandb",
        choices=["wandb", "stdout"],
        help="Log location",
    )
    parser.add_argument(
        "--target_embed_dim",
        "-t_e_d",
        type=int,
        default=64,
        help="Target embedding dimension size.",
    )
    parser.add_argument(
        "--dropout",
        "-drop",
        type=float,
        default=0.3,
        help="General dropout rate.",
    )
    parser.add_argument(
        "--teacher_forcing_ratio",
        "-tfr",
        type=float,
        default=0.6609,
        help="Probability of using teacher forcing during training.",
    )
    parser.add_argument(
        "--grad_clip_val",
        "-gcv",
        type=float,
        default=0,
        help="Value for gradient clipping (0.0 means no clipping).",
    )
    parser.add_argument(
        "--label_smoothing",
        "-ls",
        type=float,
        default=0.0564,
        help="Value for label smoothing.",
    )
    parser.add_argument(
        "--attention_type",
        "-att_type",
        type=str,
        default="bahdanau",
        choices=["bahdanau", "none"], # Added none as an option
        help="Type of attention mechanism to use ('none' for no attention).",
    )
    parser.add_argument(
        "--attention_dim",
        "-att_dim",
        type=int,
        default=64,
        help="Dimension for the attention mechanism.",
    )

    args = parser.parse_args()
    train(args)