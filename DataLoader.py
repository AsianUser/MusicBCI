import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
<<<<<<< Updated upstream
=======
import matplotlib.pyplot as plt
>>>>>>> Stashed changes


class EEGPromptWindowDataset(Dataset):


    def __init__(
        self,
        csv_path: str,
        sample_rate: int = 300,
        marker_col: str = "Trigger",
<<<<<<< Updated upstream
        pre_seconds: float = 0.0,
        post_seconds: float = 1.5,
        response_start_seconds: float = 0.2,
=======
        chunk_seconds: float = 1.5,
        skip_after_prompt_seconds: float = 0.25,
>>>>>>> Stashed changes
        label_from_trigger: bool = True,
        normalize: bool = True,
    ):
        df = pd.read_csv(csv_path, comment="#")

<<<<<<< Updated upstream
        # Columns that are not EEG channels
=======
        # Excluding Columns that are not EEG channels
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
        # Noclue what tf CHATGPT did here Optional channel-wise z-score normalization
=======
        # MAY WANT TO COMMENT NORMALIZATION OUT Noclue what tf CHATGPT did here Optional channel-wise z-score normalization
>>>>>>> Stashed changes
        if normalize:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            data = (data - mean) / std

        self.data = data
        self.trigger = trigger

<<<<<<< Updated upstream
        # Detect prompt onset: nonzero sample preceded by zero
=======
        # Detecting prompt onset from the trigger column: nonzero sample preceded by zero
>>>>>>> Stashed changes
        onsets = np.where((trigger != 0) & np.r_[True, trigger[:-1] == 0])[0]

        self.onsets = onsets.tolist()
        self.labels = [int(trigger[i]) for i in self.onsets] if label_from_trigger else [0] * len(self.onsets)

<<<<<<< Updated upstream
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
=======
        # trial windowing
        self.chunk_samples = int(round(chunk_seconds * sample_rate))
        self.skip_samples = int(round(skip_after_prompt_seconds * sample_rate))

        self.onsets = []
        self.labels = []

        # loop over tigger onset one by one
        for idx, i in enumerate(onsets):
            start = i + self.skip_samples
            end = start + self.chunk_samples

            # stop if the chunk would run past the recording
            if end > len(self.data):
                continue

            # do not allow another trigger inside this chunk
            next_onset = onsets[idx + 1] if idx + 1 < len(onsets) else len(self.data)
            if end > next_onset:
                continue
            # keeping only the valid chunks
            self.onsets.append(i)
            self.labels.append(int(trigger[i]) - 1 if label_from_trigger else 0)

    def __len__(self):
        return len(self.onsets)

    def __getitem__(self, idx):
        onset = self.onsets[idx]
        start = onset + self.skip_samples
        end = start + self.chunk_samples

        #Slide dataset
        x = self.data[start:end]  # [time, channels]
        x = torch.from_numpy(x.T.copy()).float()  # [channels, time]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
>>>>>>> Stashed changes

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




<<<<<<< Updated upstream
import torch

=======
>>>>>>> Stashed changes
csv_path = r"C:\Users\arthu\Downloads\MusicBCI_musicheadphone_TamaraRicha_PsychoBen_01_raw.csv"

loader, dataset = make_loader(
    csv_path,
    batch_size=4,
    sample_rate=300,
<<<<<<< Updated upstream
    response_start_seconds=0.25,
    post_seconds=1.5,
=======
    chunk_seconds=1.5,
    skip_after_prompt_seconds=0.25,
>>>>>>> Stashed changes
    normalize=True
)


#Everyday we shuffling/Splitting
from torch.utils.data import random_split

dataset_size = len(dataset)
<<<<<<< Updated upstream
=======
#Splitting into training and testing
>>>>>>> Stashed changes
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size

train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
<<<<<<< Updated upstream
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
=======
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)


def visualize_prompts(dataset, seconds=20, channel_idx=0):

    n_samples = min(int(seconds * dataset.sample_rate), len(dataset.data))
    t = np.arange(n_samples) / dataset.sample_rate

    plt.figure(figsize=(14, 5))
    plt.plot(t, dataset.data[:n_samples, channel_idx], label=dataset.eeg_cols[channel_idx])

    for onset in dataset.onsets:
        if onset < n_samples:
            plt.axvline(onset / dataset.sample_rate, color="red", linestyle="--", alpha=0.7)

    plt.title(f"First {seconds} seconds with prompt markers")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# To show a full trial instead of the full recording
def visualize_trial(dataset, trial_idx=0, channels_to_plot=None):
    trial_idx = min(trial_idx, len(dataset) - 1)

    onset = dataset.onsets[trial_idx]
    start = onset + dataset.skip_samples
    end = start + dataset.chunk_samples

    x = dataset.data[start:end]  # [time, channels]
    t = np.arange(x.shape[0]) / dataset.sample_rate

    if channels_to_plot is None:
        channels_to_plot = list(range(min(5, x.shape[1])))

    plt.figure(figsize=(14, 6))

    offset = 0.0
    spacing = 3.0
    for ch in channels_to_plot:
        plt.plot(t, x[:, ch] + offset, label=dataset.eeg_cols[ch])
        offset += spacing

    plt.title(f"Trial {trial_idx} | label={dataset.labels[trial_idx]}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude + offset")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

### VISUALIZING
visualize_prompts(dataset, seconds=120, channel_idx=0)
visualize_trial(dataset, trial_idx=0)
>>>>>>> Stashed changes
