from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class BlokusDataset(Dataset):
    """Blokus tensor dataset."""

    def __init__(self, csv_file, channel_index=None, transform=None):
        """

        :param csv_file:
        :param channel_index:
        :param transform:
        """
        self.tensor_lable = pd.read_csv(csv_file)
        self.transform = transform
        self.channel_index = channel_index

    def __len__(self):
        return len(self.tensor_lable)

    def __getitem__(self, item):
        tensor_name = self.tensor_lable.ix[item, 0]
        tensor = np.load(tensor_name)
        if self.channel_index:
            tensor_filter = tensor[self.channel_index, :, :]
        else:
            tensor_filter = tensor
        label = int(self.tensor_lable.ix[item, 5])

        sample = (tensor_filter, label, tensor)

        if self.transform:
            sample = self.transform(sample)

        return sample
