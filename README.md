# da6401_assignment3
This repository contains code to build and train recurrent neural networks using PyTorch.

The repository is divided into two parts - rnn and attention_rnn.

## RNN
This contains the code to build and train a Vanilla RNN transliteration model from scratch with the Dakshina Dataset. It provides a train.py file to run the code along with command-line parameters that can be passed for further configuration.

## Attention RNN (Part B)
This contains the code to build and train an Attention based RNN transliteration model from scratch with the Dakshina Dataset. It provides a train.py file to run the code along with command-line parameters that can be passed for further configuration.

## Usage
Run the training script `train.py` with the desired command-line arguments. The `train.py` script is expected to be present within the `rnn` and `attention_rnn` directories respectively. Here's an example command (adjust parameters as needed for each part):

```
python3 train.py --wandb_entity myname --wandb_project myprojectname --epochs 20 --batch_size 128 --learning_rate 0.0001 --layers 3 --cell gru --hidden_size 256 --input_embedding_size 128 --target_embed_dim 128 --bidirectional 1 --encoder_dropout 0.3 --decoder_dropout 0.3 --dropout 0.2 --teacher_forcing_ratio 0.7 --grad_clip_val 1.0 --label_smoothing 0.1 --attention_type bahdanau --attention_dim 64 --dataset_path '/projects/data/astteam/repos/da6401_sparsh/da6401_assignment3/data/dakshina_dataset_v1.0/hi/lexicons' --log_location wandb
```

## Command-Line Arguments
- `--wandb_entity` (`-we`): Wandb entity used to track experiments. Default: `myname`
- `--wandb_project` (`-wp`): Project name in Weights & Biases. Default: `myprojectname`
- `--epochs` (`-e`): Number of epochs to train. Default: `20`
- `--batch_size` (`-b`): Batch size. Default: `128` (Note: Example script uses `512` for vanilla RNN and `128` for attention RNN as defaults, adjust as per your specific script defaults)
- `--learning_rate` (`-lr`): Learning rate. Default: `1e-5` (Note: Example script uses `4e-4` for vanilla RNN and `1e-5` for attention RNN as defaults)
- `--layers` (`-l`): Number of stacked RNN layers. Default: `3`
- `--cell` (`-c`): RNN cell to be used. Choices: `rnn`, `lstm`, `gru`. Default: `lstm` (Note: Example script uses `rnn` for vanilla RNN and `lstm` for attention RNN as defaults)
- `--hidden_size` (`-h_s`): Size of RNN hidden layer. Default: `128` (Note: Example script uses `256` for vanilla RNN and `128` for attention RNN as defaults)
- `--input_embedding_size` (`-i_e_s`): Size of input embedding for RNN. Default: `128` (Note: Example script uses `256` for vanilla RNN and `128` for attention RNN as defaults)
- `--target_embed_dim` (`-t_e_d`): Target embedding dimension size. Default: `64`
- `--bidirectional` (`-bd`): Whether RNN is bidirectional or not (0 for False, 1 for True). Choices: `0`, `1`. Default: `1`
- `--encoder_dropout` (`-e_d`): Dropout probability for encoder layers. Default: `0.3` (Note: Example script uses `0.5` for vanilla RNN and `0.3` for attention RNN as defaults)
- `--decoder_dropout` (`-d_d`): Dropout probability for decoder layers. Default: `0.3` (Note: Example script uses `0.5` for vanilla RNN and `0.3` for attention RNN as defaults)
- `--dropout` (`-drop`): General dropout rate (if used elsewhere in the model). Default: `0.3`
- `--teacher_forcing_ratio` (`-tfr`): Probability of using teacher forcing during training. Default: `0.5`
- `--grad_clip_val` (`-gcv`): Value for gradient clipping (0.0 means no clipping). Default: `1.0`
- `--label_smoothing` (`-ls`): Value for label smoothing. Default: `0.1`
- `--attention_type` (`-att_type`): Type of attention mechanism to use (primarily for Attention RNN). Choices: `bahdanau`, `none`. Default: `bahdanau`
- `--attention_dim` (`-att_dim`): Dimension for the attention mechanism (if attention is used). Default: `64`
- `--dataset_path` (`-d_p`): Dataset path to the language folder (e.g., lexicons). Default: `/projects/data/astteam/repos/da6401_sparsh/da6401_assignment3/data/dakshina_dataset_v1.0/hi/lexicons`
- `--log_location` (`-g`): Log location. Choices: `wandb`, `stdout`. Default: `wandb`

**Note**: Some parameters, like `attention_type` and `attention_dim`, are primarily relevant for the Attention RNN part. For the Vanilla RNN, these might be ignored or should be set to a value indicating no attention (e.g., `--attention_type none`). Always refer to the specific `train.py` script in each part for the exact set of applicable arguments and their behavior.
