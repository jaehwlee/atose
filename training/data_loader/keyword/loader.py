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
        self.root = root
        df = pd.read_csv(
            os.path.join(self.root, "keyword", "df.csv"),
            delimiter="\t",
            names=["id", "label", "label_num", "split", "path"],
        )
        df = df[df["split"] == tr_val]
        self.input_length = input_length
        self.batch_size = batch_size
        self.fl = list(df["path"])
        self.gt = list(df["label_num"])
        self.shuffle = shuffle
        self.on_epoch_end()

    def __getitem__(self, idx):
        npy_list = []
        tag_list = []
        for i in range(self.batch_size):
            file_index = idx * self.batch_size + i
            if file_index >= len(self.fl):
                npy_list.append(np.zeros((self.input_length,)))
                tag_list.append(np.zeros((35,)))
                continue
            npy, tag = self.get_npy(file_index)
            npy_list.append(npy)
            tag_list.append(tag)

        npy_list = np.array(npy_list)
        tag_list = np.array(tag_list)
        return npy_list, tag_list

    def get_npy(self, idx):
        fn = self.fl[idx]
        npy = np.load(fn)
        if len(npy) < self.input_length:
            nnpy = np.zeros(self.input_length)
            ri = int(np.floor(np.random.random(1) * (self.input_length - len(npy))))
            nnpy[ri : ri + len(npy)] = npy
            npy = nnpy
        tag = self.gt[idx]
        tag = to_categorical(tag, num_classes=35)

        return npy, tag

    def on_epoch_end(self):
        self.indices = np.arange(len(self.fl))

    def __len__(self):
        return math.ceil(len(self.fl) / self.batch_size)


class TestLoader(Sequence):
    def __init__(
        self, root, batch_size=16, input_length=80000, tr_val="test", shuffle=False
    ):
        self.root = root
        df = pd.read_csv(
            os.path.join(self.root, "keyword", "df.csv"),
            delimiter="\t",
            names=["id", "label", "label_num", "split", "path"],
        )
        df = df[df["split"] == tr_val]
        self.fl = list(df["path"])
        self.gt = list(df["label_num"])
        self.input_length = input_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __getitem__(self, idx):
        npy_list = []
        tag_list = []
        for i in range(self.batch_size):
            file_index = idx * self.batch_size + i
            if file_index >= len(self.fl):
                npy_list.append(np.zeros((self.input_length,)))
                tag_list.append(np.zeros((35,)))
                continue
            npy, tag = self.get_npy(file_index)
            npy_list.append(npy)
            tag_list.append(tag)

        npy_list = np.array(npy_list)
        tag_list = np.array(tag_list)
        return npy_list, tag_list

    def get_npy(self, idx):
        fn = self.fl[idx]
        npy = np.load(fn)
        if len(npy) < self.input_length:
            nnpy = np.zeros(self.input_length)
            ri = int(np.floor(np.random.random(1) * (self.input_length - len(npy))))
            nnpy[ri : ri + len(npy)] = npy
            npy = nnpy
        tag = self.gt[idx]
        tag = to_categorical(tag, num_classes=35)

        return npy, tag

    def on_epoch_end(self):
        self.indices = np.arange(len(self.fl))

    def __len__(self):
        return math.ceil(len(self.fl) / self.batch_size)
