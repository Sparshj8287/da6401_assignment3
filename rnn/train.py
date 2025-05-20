import argparse
import random
import uuid
from typing import Tuple

import lightning as L
import numpy as np
import torch
import torchmetrics
from dataset import DakshinaDataModule
from decoder import Decoder
from encoder import Encoder
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from word_accuracy import WordAccuracy

import wandb


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
        Recurrent Neural Network (RNN) model for sequence-to-sequence tasks.

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
        self.encoder = Encoder(
            cell,
            hidden_size,
            n_layers,
            bidirectional,
            input_embedding_size,
            enc_dropout,
        )
        self.decoder = Decoder(
            cell,
            hidden_size,
            n_layers,
            bidirectional,
            input_embedding_size,
            dec_dropout,
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=self.decoder.vocab.index("<PAD>"))
        self.tf_prob = 100
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
        Defines the forward pass of the RNN model.

        Args:
            input_seqs (torch.Tensor): Input sequences for the encoder.
            tf_input_seqs (torch.Tensor, optional): Target sequences for teacher forcing in the decoder.
                                                    Defaults to None.
            tf_prob (float, optional): Probability (0-100) of using teacher forcing.
                                       Defaults to 0 (no teacher forcing).
        """
        hiddens = self.encoder(input_seqs)

        if tf_prob and random.randrange(1, 100) <= tf_prob:
            y = self.decoder(hiddens, tf_input_seqs)
        else:
            y = self.decoder(hiddens)

        return y

    def add_softmax_padding(self, t: torch.Tensor, c: int):
        """
        Pads a tensor `t` with softmax distributions that represent <PAD> tokens.
        This is used to make tensor `t` have the same sequence length as target sequences if `t` is shorter.

        Args:
            t (torch.Tensor): The tensor to pad (predictions). Shape (batch_size, seq_len, vocab_size).
            c (int): The number of padding steps to add.
        """
        vocab_len = len(self.decoder.vocab)
        softmax = torch.zeros(vocab_len).to(device="cuda")
        softmax[vocab_len - 3] = 1
        rep_softmax = softmax.repeat(t.shape[0], 1).unsqueeze(0).repeat(c, 1, 1)

        return torch.cat((t.transpose(0, 1), rep_softmax)).transpose(0, 1).contiguous()

    def pre_process_text(self, x: Tuple[str], y: Tuple[str]):
        """
        Converts raw text sequences (source and target) into padded integer tensors.

        Args:
            x (Tuple[str]): A batch of source text sequences.
            y (Tuple[str]): A batch of target text sequences.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Padded input sequences and padded target sequences for teacher forcing.
        """
        input_seqs = []
        tf_input_seqs = []
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

        padded_input_seqs = pad_sequence(
            input_seqs, batch_first=True, padding_value=len(self.encoder.vocab) - 1
        ).to(device="cuda")
        padded_tf_input_seqs = pad_sequence(
            tf_input_seqs, batch_first=True, padding_value=len(self.decoder.vocab) - 3
        ).to(device="cuda")

        return (padded_input_seqs, padded_tf_input_seqs)

    def training_step(self, batch, batch_idx: int):
        """
        Performs a single training step.

        Args:
            batch: A batch of data from the DataLoader (source sequences, target sequences).
            batch_idx (int): The index of the current batch.
        """
        x, y = batch

        input_seqs, tf_input_seqs = self.pre_process_text(x, y)

        y_hat = self(input_seqs, tf_input_seqs, self.tf_prob)
        diff = y_hat.shape[1] - len(tf_input_seqs[0])

        if diff < 0:
            y_hat = self.add_softmax_padding(y_hat, -diff)
        elif diff > 0:
            padding = torch.zeros(diff, len(batch[0])).to(
                device="cuda"
            ) + self.decoder.vocab.index("<PAD>")
            tf_input_seqs = torch.concat(
                (
                    tf_input_seqs.T,
                    padding,
                ),
            ).T.contiguous()

        tf_input_seqs = tf_input_seqs.to(torch.long)

        loss = self.loss(y_hat.view(-1, y_hat.size(-1)), tf_input_seqs.view(-1))

        y_hat_seq = torch.argmax(y_hat, axis=-1)

        self.train_acc(y_hat_seq, tf_input_seqs)
        self.word_train_acc(y_hat_seq, tf_input_seqs)
        self.log(
            "train_acc",
            self.train_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=len(x),
        )
        self.log(
            "word_train_acc",
            self.word_train_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=len(x),
        )
        self.log(
            "train_loss",
            loss,
            logger=True,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=len(x),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step.

        Args:
            batch: A batch of data from the DataLoader.
            batch_idx (int): The index of the current batch.
        """
        x, y = batch

        input_seqs, tf_input_seqs = self.pre_process_text(x, y)

        y_hat = self(input_seqs)

        diff = y_hat.shape[1] - len(tf_input_seqs[0])
        if diff < 0:
            y_hat = self.add_softmax_padding(y_hat, -diff)
        elif diff > 0:
            tf_input_seqs = torch.concat(
                (
                    tf_input_seqs.T,
                    torch.zeros(diff, len(batch[0])).to(device="cuda")
                    + self.decoder.vocab.index("<PAD>"),
                ),
            ).T.contiguous()

        tf_input_seqs = tf_input_seqs.to(torch.long)

        loss = self.loss(y_hat.view(-1, y_hat.size(-1)), tf_input_seqs.view(-1))

        y_hat_seq = torch.argmax(y_hat, axis=-1)

        # decoder_vocab = np.array(self.decoder.vocab)

        # y_hat_word = np.array(
        #     [
        #         "".join(decoder_vocab[word].tolist())
        #         for word in y_hat_seq.detach().cpu().numpy()
        #     ]
        # )
        # y_vec_word = np.array(
        #     [
        #         "".join(decoder_vocab[word].tolist())
        #         for word in tf_input_seqs.detach().cpu().numpy()
        #     ]
        # )

        # print(np.array(list(zip(y_hat_word, y_vec_word)))[:5])

        self.val_acc(y_hat_seq, tf_input_seqs)
        self.word_val_acc(y_hat_seq, tf_input_seqs)
        self.log(
            "val_acc",
            self.val_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=len(x),
        )
        self.log(
            "word_val_acc",
            self.word_val_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(x),
        )
        self.log(
            "val_loss",
            loss,
            logger=True,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(x),
        )

    def on_train_epoch_end(self) -> None:
        self.tf_prob = 100 - self.trainer.current_epoch * 100 / self.trainer.max_epochs

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def train(args):
    """
    Main training function. Sets up logging, data, model, and trainer, then starts training.

    Args:
        args (argparse.Namespace): Command-line arguments and hyperparameters.
    """
    if args.log_location == "wandb":
        wandb.init(entity=args.wandb_entity, project=args.wandb_project)
        wandb.run.name = f"cell_{args.cell}_bd_{args.bidirectional}_bs_{args.batch_size}_l_{args.layers}_e_{args.epochs}_hs_{args.hidden_size}_ies_{args.input_embedding_size}_id_{wandb.run.id}"
        model_name = wandb.run.name
        wandb_logger = WandbLogger(name=args.wandb_entity, log_model=False)
    else:
        model_name = str(uuid.uuid4())
        wandb_logger = None

    print("Training with the following hyperparameters:")
    print(args)

    dakshina = DakshinaDataModule(args.dataset_path, args.batch_size)

    model = RNN(
        args.layers,
        args.cell,
        args.hidden_size,
        args.input_embedding_size,
        args.learning_rate,
        bool(args.bidirectional),
        args.encoder_dropout,
        args.decoder_dropout,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=model_name,
        save_top_k=1,
        verbose=True,
        monitor="val_acc",
        mode="max",
        # prefix=,
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        logger=wandb_logger,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model=model, datamodule=dakshina)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an RNN based transliteration model"
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
    parser.add_argument("--batch_size", "-b", type=int, default=256, help="Batch size.")
    parser.add_argument(
        "--learning_rate", "-lr", type=float, default=0.0036, help="Learning rate."
    )
    parser.add_argument(
        "--layers", "-l", type=int, default=3, help="Number of stacked RNN Layers"
    )
    parser.add_argument(
        "--cell",
        "-c",
        type=str,
        default="lstm",
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
        default=32,
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
        default=0.1131,
        help="Dropout for encoder layers",
    )
    parser.add_argument(
        "--decoder_dropout",
        "-d_d",
        type=float,
        default=0.1131,
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
        default=128,
        help="Target embedding dimension size.",
    )
    parser.add_argument(
        "--dropout",
        "-drop",
        type=float,
        default=0.3,
        help="General dropout rate (if used elsewhere in the model).",
    )
    parser.add_argument(
        "--teacher_forcing_ratio",
        "-tfr",
        type=float,
        default=0.4331,
        help="Probability of using teacher forcing during training.",
    )
    parser.add_argument(
        "--grad_clip_val",
        "-gcv",
        type=float,
        default=5.0,
        help="Value for gradient clipping (0.0 means no clipping).",
    )
    parser.add_argument(
        "--label_smoothing",
        "-ls",
        type=float,
        default=0.1632,
        help="Value for label smoothing.",
    )

    args = parser.parse_args()
    
    # train(args) # Ensure your train function is defined and uncomment this line
    
    print("Parsed arguments:")
    for arg_name, arg_value in vars(args).items():
        print(f"  {arg_name}: {arg_value}")


    train(args)