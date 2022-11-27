import numpy as np
import math
from tensorflow.keras.utils import Sequence
import os
import pandas as pd

np.random.seed(42)
from tensorflow.keras.utils import to_categorical


class TrainLoader(Sequence):
    def __init__(
        self, root, batch_size=16, input_length=80000, tr_val="train", shuffle=False
    ):
        df = pd.read_csv(
            os.path.join(root, "dcase", "df.csv"),
            delimiter="\t",
            names=["file", "start", "end", "path", "split", "label"],
        )
        self.root = root
        df = df[df["split"] == tr_val]
        self.input_length = input_length
        self.batch_size = batch_size
        self.fl = list(df["path"])
        self.binary = list(df["label"])
        self.shuffle = shuffle
        self.on_epoch_end()

    def __getitem__(self, idx):
        npy_list = []
        tag_list = []
        for i in range(self.batch_size):
            file_index = idx * self.batch_size + i
            if file_index >= len(self.fl):
                npy_list.append(np.zeros((self.input_length,)))
                tag_list.append(np.zeros((17,)))
                continue
            npy, tag = self.get_npy(file_index)
            npy_list.append(npy)
            tag_list.append(tag)

        npy_list = np.array(npy_list)
        tag_list = np.array(tag_list)
        return npy_list, tag_list

    def get_npy(self, idx):
        fn = self.fl[idx]
        npy = np.load(fn, mmap_mode="r")
        if len(npy) < self.input_length:
            nnpy = np.zeros(self.input_length)
            ri = int(np.floor(np.random.random(1) * (self.input_length - len(npy))))
            nnpy[ri : ri + len(npy)] = npy
            npy = nnpy
        random_idx = int(np.floor(np.random.random(1) * (len(npy) - self.input_length)))
        npy = np.array(npy[random_idx : random_idx + self.input_length])
        tag = np.fromstring(self.binary[idx][1:-1], dtype=np.float32, sep=" ")
        return npy, tag

    def on_epoch_end(self):
        self.indices = np.arange(len(self.fl))
        # if self.shuffle == True:
        # np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self.fl) / self.batch_size)


class TestLoader(Sequence):
    def __init__(
        self, root, batch_size=16, input_length=80000, tr_val="test", shuffle=False
    ):
        self.root = root
        df = pd.read_csv(
            os.path.join(self.root, "dcase", "df.csv"),
            delimiter="\t",
            names=["file", "start", "end", "path", "split", "label"],
        )
        df = df[df["split"] == tr_val]
        self.fl = list(df["path"])
        self.binary = list(df["label"])
        self.input_length = input_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __getitem__(self, idx):
        npy_list = []
        tag_list = []
        npy, tag = self.get_npy(idx)
        hop = (len(npy) - self.input_length) // self.batch_size

        for i in range(self.batch_size):
            if self.input_length > len(npy):
                npy_list.append(
                    np.zeros(
                        self.input_length,
                    )
                )
                tag_list.append(
                    np.zeros(
                        17,
                    )
                )
                continue
            x = np.array(npy[i * hop : i * hop + self.input_length])
            npy_list.append(x)
            tag_list.append(tag)

        npy_list = np.array(npy_list)
        tag_list = np.array(tag_list)
        return npy_list, tag_list

    def get_npy(self, idx):
        fn = self.fl[idx]
        npy = np.load(fn, mmap_mode="r")
        tag = np.fromstring(self.binary[idx][1:-1], dtype=np.float32, sep=" ")

        return npy, tag

    def on_epoch_end(self):
        self.indices = np.arange(len(self.fl))
        # if self.shuffle == True:
        # np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.fl)
