"""Realistic multimodal dataset: MNIST images + text descriptions.

This creates a more challenging task where:
- Images are actual 28x28 MNIST digits (flattened to 784 dims, then reduced)
- Text describes attributes of the digit (color, size, style, etc.)
- The label depends on BOTH image and text (e.g., "small odd digit" vs "large even digit")

Neither modality alone can determine the class - you need both.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNISTTextDataset(Dataset):
    """MNIST images + synthetic text descriptions.
    
    Task: Classify based on digit AND text attributes.
    Classes:
    0: small (0-4) + odd
    1: small (0-4) + even  
    2: large (5-9) + odd
    3: large (5-9) + even
    4: small + prime (2,3,5,7 but 5,7 are large, so 2,3)
    5: large + composite (4,6,8,9)
    6: odd + less than 6
    7: even + greater than 5
    8: small + not 1
    9: large + not 9
    
    Text features encode: {small/large, odd/even, prime/composite, <6/>5, is_1, is_9}
    
    This creates a task where you MUST look at both image (what digit is it?)
    AND text (what attributes does it have?) to classify correctly.
    """
    
    def __init__(self, root='./data', train=True, text_dim=32, noise=0.1, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Load MNIST
        self.mnist = datasets.MNIST(root=root, train=train, download=True,
                                     transform=transforms.ToTensor())
        
        self.text_dim = text_dim
        self.noise = noise
        
        # Pre-process all samples
        self.images = []
        self.texts = []
        self.labels = []
        
        for idx in range(len(self.mnist)):
            img, digit = self.mnist[idx]
            img_flat = img.view(-1)  # 784 dims
            
            # Determine class based on digit properties
            is_small = digit < 5
            is_large = digit >= 5
            is_odd = digit % 2 == 1
            is_even = digit % 2 == 0
            is_prime = digit in [2, 3, 5, 7]
            is_composite = digit in [4, 6, 8, 9]
            is_1 = digit == 1
            is_9 = digit == 9
            
            # Assign to class (requires both digit ID and attributes)
            if is_small and is_odd:
                label = 0
            elif is_small and is_even:
                label = 1
            elif is_large and is_odd:
                label = 2
            elif is_large and is_even:
                label = 3
            elif is_small and is_prime:  # only 2,3
                label = 4
            elif is_large and is_composite:  # 6,8,9 (5,7 are prime)
                label = 5
            elif is_odd and digit < 6:
                label = 6
            elif is_even and digit > 5:
                label = 7
            elif is_small and not is_1:
                label = 8
            elif is_large and not is_9:
                label = 9
            else:
                # Shouldn't reach here, but fallback
                label = digit % 10
            
            # Create text features that encode the attributes
            # The text describes properties, not the digit itself
            text_features = np.zeros(6, dtype=np.float32)
            text_features[0] = 1.0 if is_small else -1.0  # small vs large
            text_features[1] = 1.0 if is_odd else -1.0    # odd vs even
            text_features[2] = 1.0 if is_prime else -1.0  # prime vs composite
            text_features[3] = 1.0 if digit < 6 else -1.0  # <6 vs >5
            text_features[4] = 1.0 if is_1 else 0.0        # is 1
            text_features[5] = 1.0 if is_9 else 0.0        # is 9
            
            # Expand to text_dim with noise
            text_full = np.random.normal(0, noise, size=text_dim).astype(np.float32)
            text_full[:6] = text_features
            
            # Add some random text features that correlate with digit
            # This creates subtle patterns that help but aren't decisive
            text_full[6:12] = np.random.normal(digit / 10.0, 0.1, size=6)
            
            # Store
            self.images.append(img_flat)
            self.texts.append(torch.tensor(text_full))
            self.labels.append(label)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.texts[idx], self.labels[idx]
    

class CorruptedMNISTTextDataset(Dataset):
    """Wrapper that applies corruption to MNISTTextDataset."""
    
    def __init__(self, base_dataset, corruption_type='clean', corruption_level=0.5):
        self.base = base_dataset
        self.corruption_type = corruption_type
        self.corruption_level = corruption_level
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        img, text, label = self.base[idx]
        
        if self.corruption_type == 'clean':
            pass
        elif self.corruption_type == 'noisy_image':
            # Add Gaussian noise to image
            img = img + self.corruption_level * torch.randn_like(img)
            img = torch.clamp(img, 0, 1)
        elif self.corruption_type == 'missing_image':
            # Zero out image
            img = torch.zeros_like(img)
        elif self.corruption_type == 'text_dropout':
            # Randomly zero out text features
            mask = (torch.rand_like(text) > self.corruption_level).float()
            text = text * mask
        elif self.corruption_type == 'missing_text':
            # Zero out all text
            text = torch.zeros_like(text)
        elif self.corruption_type == 'wrong_text':
            # Flip some text features (adversarial)
            noise = torch.randn_like(text) * self.corruption_level
            text = text + noise
        elif self.corruption_type == 'occluded_image':
            # Randomly occlude parts of image (images are flattened 784-dim = 28x28)
            mask = torch.ones_like(img)
            x, y = np.random.randint(0, 14, size=2)
            # Set 14x14 patch to zero (196 pixels)
            for i in range(14):
                for j in range(14):
                    idx = (x + i) * 28 + (y + j)
                    if idx < 784:
                        mask[idx] = 0
            img = img * mask
        
        return img, text, label
