import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticMultimodalDataset(Dataset):
    """Tiny synthetic image/text dataset for TCE-Lite tests.

    Each sample has:
    - image vector: contains partial information (coarse class)
    - text vector: contains partial information (fine class)
    - label: requires BOTH modalities to determine (neither alone is sufficient)
    
    The label is split: image encodes (label // 2), text encodes (label % 2).
    This creates 5 coarse classes from image and 2 fine classes from text,
    combining to produce 10 final classes. Neither modality alone can 
    determine the label - you need both.
    """

    def __init__(self, n_samples=5000, image_dim=64, text_dim=32, n_classes=10, noise=0.05, seed=42):
        assert n_classes == 10, "This dataset requires exactly 10 classes for the split design"
        rng = np.random.default_rng(seed)
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.n_classes = n_classes

        labels = rng.integers(0, n_classes, size=n_samples)
        image = rng.normal(0, noise, size=(n_samples, image_dim)).astype(np.float32)
        text = rng.normal(0, noise, size=(n_samples, text_dim)).astype(np.float32)

        for i, y in enumerate(labels):
            # Split the label: image gets coarse info (which of 5 groups), text gets fine info (which of 2)
            coarse = y // 2  # 0-4 (5 coarse classes)
            fine = y % 2     # 0-1 (2 fine classes)
            
            # Image encodes coarse class with two features
            image[i, coarse % image_dim] += 1.0
            image[i, (coarse * 7 + 3) % image_dim] += 0.5
            
            # Text encodes fine class with two features
            text[i, fine % text_dim] += 1.0
            text[i, (fine * 11 + 5) % text_dim] += 0.5
            
            # Add interaction pattern: a feature that depends on both
            # This creates correlation between modalities that helps fusion
            interaction_strength = (coarse + fine) / 6.0  # normalized
            image[i, (coarse + fine * 3) % image_dim] += interaction_strength * 0.3
            text[i, (fine + coarse * 2) % text_dim] += interaction_strength * 0.3

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
