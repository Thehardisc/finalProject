"""PyTorch Dataset for variable-length conversation sequences."""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from typing import List


class ConversationDataset(Dataset):
    def __init__(
        self,
        sequences:  List[np.ndarray],
        targets:    List[np.ndarray],
        trajectory_types: List[str],
    ):
        assert len(sequences) == len(targets)
        self.sequences        = sequences
        self.targets          = targets
        self.trajectory_types = trajectory_types

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        tgt = self.targets[idx]

        x = torch.tensor(seq[:-1], dtype=torch.float32)
        y = torch.tensor(tgt,      dtype=torch.float32)
        length = x.shape[0]

        return x, y, length, self.trajectory_types[idx]


def collate_fn(batch):
    """Pad sequences in a batch to the same length."""
    xs, ys, lengths, trajs = zip(*batch)

    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
    ys_padded = pad_sequence(ys, batch_first=True, padding_value=0.0)
    lengths   = torch.tensor(lengths, dtype=torch.long)

    return xs_padded, ys_padded, lengths, trajs


def load_sequence_dataset(features_dir: Path):
    """Load sequences.pkl and split into train/val/test by trajectory type."""
    pkl_path = features_dir / "sequences.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"{pkl_path} not found. Run feature extraction first.")

    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)

    sequences   = data["sequences"]
    seq_targets = data["seq_targets"]
    trajs       = data["seq_trajectories"]

    valid = [
        (s, t, tr)
        for s, t, tr in zip(sequences, seq_targets, trajs)
        if s.shape[0] >= 3
    ]
    if not valid:
        raise ValueError("No sequences with length ≥ 3. Generate more data.")

    sequences, seq_targets, trajs = zip(*valid)

    rng = np.random.default_rng(42)
    traj_arr = np.array(trajs)
    train_idx, val_idx, test_idx = [], [], []

    for ttype in np.unique(traj_arr):
        idx = np.where(traj_arr == ttype)[0]
        rng.shuffle(idx)
        n_train = max(1, int(len(idx) * 0.70))
        n_val   = max(0, int(len(idx) * 0.15))
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train:n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val:].tolist())

    def _make(idxs):
        if not idxs:
            return None
        return ConversationDataset(
            [sequences[i]   for i in idxs],
            [seq_targets[i] for i in idxs],
            [trajs[i]       for i in idxs],
        )

    return _make(train_idx), _make(val_idx), _make(test_idx)


def make_loaders(train_ds, val_ds, test_ds, batch_size: int = 8):
    """Create DataLoaders for each split."""
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=False,
    ) if train_ds else None

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn,
    ) if val_ds else None
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn,
    ) if test_ds else None


    return train_loader, val_loader, test_loader
