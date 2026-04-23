import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticMultimodalDataset(Dataset):
    """Tiny synthetic image/text dataset for early TCE-Lite tests.

    Each sample has:
    - image vector: noisy numeric features
    - text vector: symbolic/rule-like features
    - label: class derived from both modalities
    """

    def __init__(self, n_samples=5000, image_dim=64, text_dim=32, n_classes=10, noise=0.05, seed=42):
        rng = np.random.default_rng(seed)
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.n_classes = n_classes

        labels = rng.integers(0, n_classes, size=n_samples)
        image = rng.normal(0, noise, size=(n_samples, image_dim)).astype(np.float32)
        text = rng.normal(0, noise, size=(n_samples, text_dim)).astype(np.float32)

        for i, y in enumerate(labels):
            image[i, y % image_dim] += 1.0
            image[i, (y * 3 + 7) % image_dim] += 0.5
            text[i, y % text_dim] += 1.0
            text[i, (y * 2 + 5) % text_dim] += 0.5

        self.image = torch.tensor(image)
        self.text = torch.tensor(text)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.image[idx], self.text[idx], self.labels[idx]


def corrupt_batch(image, text, image_noise=0.0, text_dropout=0.0, missing="none"):
    if image_noise > 0:
        image = image + image_noise * torch.randn_like(image)

    if text_dropout > 0:
        mask = (torch.rand_like(text) > text_dropout).float()
        text = text * mask

    if missing == "image":
        image = torch.zeros_like(image)
    elif missing == "text":
        text = torch.zeros_like(text)

    return image, text
