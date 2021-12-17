# Create torch dataset

import torch


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        # end if
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
