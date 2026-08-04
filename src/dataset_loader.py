import torch
from torch.utils.data import Dataset
import numpy as np

class RoadDataset(Dataset):
    def __init__(
        self,
        recordings,
        window_size=128,
        stride=32,
        has_labels=False,
    ):
        self.recordings = recordings
        self.window_size = window_size
        self.has_labels = has_labels

        self.indices = []

        for rec_idx, rec in enumerate(recordings):
            T = rec.shape[0]
            for start in range(0, T - window_size + 1, stride):
                self.indices.append((rec_idx, start))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        rec_idx, start = self.indices[idx]
        window = self.recordings[rec_idx][
            start:start+self.window_size
        ]

        if self.has_labels:
            x = window[:, :-1].astype(np.float32)
            y = window[:, -1]
            action = x[-1, 0]

            label = int(y.mean() > 0.5)
            # label = y[-1]

            return (
                torch.from_numpy(x),
                torch.tensor(label, dtype=torch.long), torch.tensor(action)
            )

        else:
            x = window.astype(np.float32)
            return torch.from_numpy(x)