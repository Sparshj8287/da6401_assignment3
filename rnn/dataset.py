import os
from typing import List

import glob2
import lightning as L
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class Dakshina(Dataset):
    def __init__(self, root: str, stage: str):
        """
        Args:
            root (str): The root directory where the dataset is stored.
                        Subdirectories within this root are considered as labels.
            stage (str): The stage of dataset usage, e.g., "fit", "valid", "test".
        """
        self.labels: List[str] = next(os.walk(root))[1] # Assumes subdirectories in root are the labels
        self.data = self.__load_data(root, stage)

    def __load_data(self, root: str, stage: str):
        """
        Loads data from .tsv files based on the specified root directory and stage.

        Args:
            root (str): The root directory containing the data files.
            stage (str): The stage identifier ("fit", "valid", "test", or other for predict).
        """
        # Determine file patterns based on the stage (fit, valid, test)
        if stage == "fit":
            paths = glob2.glob(f"{root}/*train.tsv")
        elif stage == "valid":
            paths = glob2.glob(f"{root}/*dev.tsv")
        elif stage == "test":
            paths = glob2.glob(f"{root}/*test.tsv")
        else: # Default to test if stage is not recognized (e.g., for 'predict')
            paths = glob2.glob(f"{root}/*test.tsv")

        li = []

        for path in paths:
            df = pd.read_csv(path, header=None, sep="\t")
            li.append(df)

        if not li:
            # Provide a warning if no data files are found for the specified stage
            stage_suffix = "train" if stage == "fit" else "dev" if stage == "valid" else "test"
            print(f"Warning: No files found for stage '{stage}' with pattern '{root}/*{stage_suffix}.tsv'. Returning empty DataFrame.")
            return pd.DataFrame(columns=[0, 1, 2]) # Return empty DataFrame with expected columns

        # Concatenate multiple dataframes if found, otherwise use the single one
        if len(li) > 1:
            frame = pd.concat(li, axis=0, ignore_index=True)
        else:
            frame = li[0]

        # Ensure specified columns are treated as strings
        if 0 in frame.columns:
            frame[0] = frame[0].astype(str)
        if 1 in frame.columns:
            frame[1] = frame[1].astype(str)

        return frame

    def __len__(self):
        # No arguments other than self
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Retrieves an item from the dataset at the given index.

        Args:
            idx (int): The index of the data item to retrieve.
        """
        # Returns a tuple (item_from_column_1, item_from_column_0)
        # Typically (target, source) or (label, input_text)
        return self.data[1][idx], self.data[0][idx]


class DakshinaDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        """
        Args:
            data_dir (str): The directory where the data is located.
            batch_size (int): The batch size to be used by DataLoaders.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        # Variables to hold datasets for different stages
        self.dakshina_full = None
        self.dakshina_val = None
        self.dakshina_test = None
        self.dakshina_predict = None


    def setup(self, stage: str):
        """
        Sets up the datasets for the specified stage.

        Args:
            stage (str): The current operational stage, e.g., "fit", "test", "predict".
                         This determines which datasets are initialized.
        """
        # Setup datasets based on the current stage (fit, test, predict)
        if stage == "fit":
            self.dakshina_full = Dakshina(self.data_dir, "fit")
            self.dakshina_val = Dakshina(self.data_dir, "valid")
        elif stage == "test":
            self.dakshina_test = Dakshina(self.data_dir, "test")
        elif stage == "predict":
            self.dakshina_predict = Dakshina(self.data_dir, "predict") # 'predict' stage will use test data pattern

    def train_dataloader(self):
        # No arguments other than self
        return DataLoader(
            self.dakshina_full,
            batch_size=self.batch_size,
            num_workers=11, # Number of worker processes for data loading
            shuffle=True,   # Shuffle data for training
        )

    def val_dataloader(self):
        # No arguments other than self
        return DataLoader(
            self.dakshina_val,
            batch_size=self.batch_size,
            num_workers=11
        )

    def test_dataloader(self):
        # No arguments other than self
        return DataLoader(
            self.dakshina_test,
            batch_size=self.batch_size,
            num_workers=11
        )

    def predict_dataloader(self):
        # No arguments other than self
        return DataLoader(
            self.dakshina_predict,
            batch_size=self.batch_size,
            num_workers=11,
        )
