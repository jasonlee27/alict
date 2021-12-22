# Create torch dataset

import torch


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None, label_vec_len=2):
        self.encodings = encodings
        self.labels = labels
        self.label_vec_len = label_vec_len

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        if self.labels:
            if self.label_vec_len==2: # when labels consists of (neg, pos)
                label = self.labels[idx]
                if label==0:
                    label = [1.,0.]
                elif label==1:
                    label = [0.,1.]
                else:
                    label = [0.5,0.5]
                # end if
            else: # when labels consists of (neg, neu, pos)
                label = self.labels[idx]
                if label==0:
                    label = [1.,0.,0.]
                elif label==1:
                    label = [0.,1.,0.]
                else:
                    label = [0.,0.,1.]
                # end if
            item["labels"] = torch.tensor(label)
        # end if
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
