import os

import numpy as np
import pandas as pd
import requests
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class ECGDataset(Dataset):
    def __init__(
        self,
        csv_url: str,
        train=True,
        filter: str = "all",
        transform=None,
        target_transform=None,
    ):
        dataframe = pd.read_csv(
            csv_url,
            header=None,
        )
        raw_data = dataframe.values

        # The last element contains the labels
        labels = raw_data[:, -1]

        # The other data points are the electrocadriogram data
        data = raw_data[:, 0:-1]

        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, test_size=0.2, random_state=21
        )

        max_val = np.max(train_data)
        min_val = np.min(train_data)

        self.data = train_data if train else test_data
        self.labels = train_labels if train else test_labels

        self.data = (self.data - min_val) / (max_val - min_val)  # normalize

        self.data = self.data.astype(np.float32)
        self.labels = self.labels.astype(np.bool)

        self.filter = filter
        if self.filter == "normal":
            self.data = self.data[self.labels]
            self.labels = self.labels[self.labels]
        elif self.filter == "anomalous":
            self.data = self.data[~self.labels]
            self.labels = self.labels[~self.labels]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx].item()

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)

        return data, label


def load_data(batch_size: int):
    file_url = "http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv"
    local_path = "data/ecg.csv"
    if not os.path.exists(local_path):
        response = requests.get(file_url)
        with open(local_path, "wb") as f:
            f.write(response.content)

    normal_train_data = ECGDataset(
        local_path,
        train=True,
        filter="normal",
        transform=torch.from_numpy,
    )
    test_data = ECGDataset(
        local_path,
        train=False,
        filter="all",
        transform=torch.from_numpy,
    )
    normal_test_data = ECGDataset(
        local_path,
        train=False,
        filter="normal",
        transform=torch.from_numpy,
    )
    anomalous_test_data = ECGDataset(
        local_path,
        train=False,
        filter="anomalous",
        transform=torch.from_numpy,
    )

    return (
        DataLoader(
            normal_train_data,
            batch_size=batch_size,
            shuffle=True,
        ),
        DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
        ),
        DataLoader(
            normal_test_data,
            batch_size=batch_size,
            shuffle=False,
        ),
        DataLoader(
            anomalous_test_data,
            batch_size=batch_size,
            shuffle=False,
        ),
    )
