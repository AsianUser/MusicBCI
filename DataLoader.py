import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class EEGPromptWindowDataset(Dataset):


    def __init__(
        self,
        csv_path: str,
        sample_rate: int = 300,
        marker_col: str = "Trigger",
        pre_seconds: float = 0.0,
        post_seconds: float = 1.5,
        response_start_seconds: float = 0.2,
        label_from_trigger: bool = True,
        normalize: bool = True,
    ):
        df = pd.read_csv(csv_path, comment="#")

        # Columns that are not EEG channels
        exclude = {
            "Time", marker_col, "Time_Offset", "ADC_Status",
            "ADC_Sequence", "Event", "Comments"
        }
        eeg_cols = [c for c in df.columns if c not in exclude]

        # Convert numeric columns safely
        for c in eeg_cols + [marker_col]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=eeg_cols).reset_index(drop=True)

        self.eeg_cols = eeg_cols
        self.sample_rate = sample_rate
        self.marker_col = marker_col

        data = df[eeg_cols].to_numpy(dtype=np.float32)
        trigger = df[marker_col].fillna(0).astype(int).to_numpy()

        # Noclue what tf CHATGPT did here Optional channel-wise z-score normalization
        if normalize:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            data = (data - mean) / std

        self.data = data
        self.trigger = trigger

        # Detect prompt onset: nonzero sample preceded by zero
        onsets = np.where((trigger != 0) & np.r_[True, trigger[:-1] == 0])[0]

        self.onsets = onsets.tolist()
        self.labels = [int(trigger[i]) for i in self.onsets] if label_from_trigger else [0] * len(self.onsets)

        # Trial windowing
        self.pre_samples = int(round(pre_seconds * sample_rate))
        self.post_samples = int(round(post_seconds * sample_rate))
        self.response_start_samples = int(round(response_start_seconds * sample_rate))

        # final window begins after prompt + response_start_seconds
        self.window_len = max(1, self.post_samples - self.response_start_samples)

        # Keep only valid windows that have enough data
        valid = []
        for idx, onset in enumerate(self.onsets):
            start = onset + self.response_start_samples - self.pre_samples
            end = start + self.window_len
            if end <= len(self.data):
                valid.append(idx)

        self.valid_indices = valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        trial_idx = self.valid_indices[idx]
        onset = self.onsets[trial_idx]

        start = onset + self.response_start_samples - self.pre_samples
        end = start + self.window_len

        x = self.data[start:end]  # [time, channels]

        # Return as [channels, time] for PyTorch models
        x = torch.from_numpy(x.T.copy()).float()

        # Trigger values 1..8 become class labels 0..7
        y = torch.tensor(self.labels[trial_idx] - 1, dtype=torch.long)

        return x, y


def make_loader(
    csv_path: str,
    batch_size: int = 8,
    shuffle: bool = True,
    **dataset_kwargs
):
    dataset = EEGPromptWindowDataset(csv_path, **dataset_kwargs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return loader, dataset




import torch

csv_path = r"C:\Users\arthu\Downloads\MusicBCI_musicheadphone_TamaraRicha_PsychoBen_01_raw.csv"

loader, dataset = make_loader(
    csv_path,
    batch_size=4,
    sample_rate=300,
    response_start_seconds=0.25,
    post_seconds=1.5,
    normalize=True
)


#Everyday we shuffling/Splitting
from torch.utils.data import random_split

dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size

train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)