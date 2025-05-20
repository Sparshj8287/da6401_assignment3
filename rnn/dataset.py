import os
from typing import List

import glob2
import lightning as L
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class Dakshina(Dataset):
    def __init__(self, root: str, stage: str):
        self.labels: List[str] = next(os.walk(root))[1]
        self.data = self.__load_data(root, stage)

    def __load_data(self, root, stage: str):
        if stage == "fit":
            paths = glob2.glob(f"{root}/*train.tsv")
        elif stage == "valid":
            paths = glob2.glob(f"{root}/*dev.tsv")
        elif stage == "test":
            paths = glob2.glob(f"{root}/*test.tsv")
        else:
            paths = glob2.glob(f"{root}/*test.tsv")

        li = []

        for path in paths:
            df = pd.read_csv(path, header=None, sep="\t")
            li.append(df)

        if not li:
            stage_suffix = "train" if stage == "fit" else "dev" if stage == "valid" else "test"
            print(f"Warning: No files found for stage '{stage}' with pattern '{root}/*{stage_suffix}.tsv'. Returning empty DataFrame.")
            return pd.DataFrame(columns=[0, 1, 2])

        if len(li) > 1:
            frame = pd.concat(li, axis=0, ignore_index=True)
        else:
            frame = li[0]

        if 0 in frame.columns:
            frame[0] = frame[0].astype(str)
        if 1 in frame.columns:
            frame[1] = frame[1].astype(str)

        return frame

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[1][idx], self.data[0][idx]


class DakshinaDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        if stage == "fit":
            self.dakshina_full = Dakshina(self.data_dir, stage)
            self.dakshina_val = Dakshina(self.data_dir, "valid")
        elif stage == "test":
            self.dakshina_test = Dakshina(self.data_dir, stage)
        elif stage == "predict":
            self.dakshina_predict = Dakshina(self.data_dir, stage)

    def train_dataloader(self):
        return DataLoader(
            self.dakshina_full,
            batch_size=self.batch_size,
            num_workers=11,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dakshina_val, batch_size=self.batch_size, num_workers=11
        )

    def test_dataloader(self):
        return DataLoader(
            self.dakshina_test, batch_size=self.batch_size, num_workers=11
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dakshina_predict,
            batch_size=self.batch_size,
            num_workers=11,
        )
