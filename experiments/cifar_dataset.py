"""CIFAR-10 with text descriptions - harder multimodal task.

CIFAR-10 is 32x32 color images (3072 dims) with 10 classes.
Text describes visual attributes + class-related features.
Task: Classify based on image AND text attributes.

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class CIFARTextDataset(Dataset):
    """CIFAR-10 images + synthetic text descriptions.
    
    Text features describe:
    - Color attributes (warm/cool colors, bright/dark)
    - Shape/texture (smooth/textured, geometric/organic)
    - Size (small/medium/large in image)
    - Context clues (ground/water/air)
    - Class hints (not direct labels, but correlated attributes)
    
    The task requires both image and text to classify correctly.
    """
    
    def __init__(self, root='./data', train=True, text_dim=64, noise=0.1, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Load CIFAR-10
        transform = transforms.ToTensor()
        self.cifar = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        
        # CIFAR-10 classes
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Class attributes for text generation
        self.class_attributes = {
            0: {  # airplane
                'warm_color': 0.7, 'bright': 0.8, 'smooth': 0.9, 'geometric': 0.9,
                'flying': 0.9, 'artificial': 1.0, 'large': 0.6, 'in_air': 0.9
            },
            1: {  # automobile
                'warm_color': 0.5, 'bright': 0.6, 'smooth': 0.8, 'geometric': 0.7,
                'ground': 0.9, 'artificial': 1.0, 'medium': 0.7, 'wheels': 0.9
            },
            2: {  # bird
                'warm_color': 0.3, 'bright': 0.4, 'textured': 0.7, 'organic': 0.9,
                'flying': 0.7, 'natural': 0.9, 'small': 0.7, 'in_air': 0.6
            },
            3: {  # cat
                'warm_color': 0.4, 'bright': 0.3, 'textured': 0.6, 'organic': 0.9,
                'ground': 0.8, 'natural': 0.8, 'small': 0.6, 'furry': 0.8
            },
            4: {  # deer
                'warm_color': 0.5, 'bright': 0.5, 'textured': 0.6, 'organic': 0.9,
                'ground': 0.9, 'natural': 0.9, 'medium': 0.7, 'legs': 0.9
            },
            5: {  # dog
                'warm_color': 0.5, 'bright': 0.5, 'textured': 0.7, 'organic': 0.9,
                'ground': 0.9, 'natural': 0.8, 'medium': 0.6, 'furry': 0.8
            },
            6: {  # frog
                'cool_color': 0.7, 'bright': 0.6, 'smooth': 0.6, 'organic': 0.8,
                'ground': 0.6, 'natural': 0.9, 'small': 0.7, 'water': 0.6
            },
            7: {  # horse
                'warm_color': 0.6, 'bright': 0.5, 'textured': 0.5, 'organic': 0.9,
                'ground': 0.9, 'natural': 0.9, 'large': 0.7, 'legs': 0.9
            },
            8: {  # ship
                'cool_color': 0.6, 'bright': 0.5, 'smooth': 0.7, 'geometric': 0.8,
                'water': 0.9, 'artificial': 0.9, 'large': 0.8, 'horizontal': 0.7
            },
            9: {  # truck
                'warm_color': 0.4, 'bright': 0.5, 'smooth': 0.6, 'geometric': 0.8,
                'ground': 0.9, 'artificial': 1.0, 'large': 0.8, 'boxy': 0.8
            }
        }
        
        self.text_dim = text_dim
        self.noise = noise
        
        # Pre-generate text features for all samples
        self.images = []
        self.texts = []
        self.labels = []
        
        for idx in range(len(self.cifar)):
            img, label = self.cifar[idx]
            img_flat = img.view(-1)  # 3072 dims (3x32x32)
            
            # Generate text from class attributes + noise
            attrs = self.class_attributes[label]
            text_features = np.zeros(text_dim, dtype=np.float32)
            
            # Fill first 16 dims with semantic attributes
            text_features[0] = attrs.get('warm_color', 0) - attrs.get('cool_color', 0)
            text_features[1] = attrs.get('bright', 0.5) * 2 - 1
            text_features[2] = attrs.get('smooth', 0) - attrs.get('textured', 0)
            text_features[3] = attrs.get('geometric', 0) - attrs.get('organic', 0)
            text_features[4] = attrs.get('flying', 0) * 2 - 1
            text_features[5] = attrs.get('ground', 0) * 2 - 1
            text_features[6] = attrs.get('water', 0) * 2 - 1
            text_features[7] = attrs.get('artificial', 0) * 2 - 1
            text_features[8] = attrs.get('natural', 0) * 2 - 1
            text_features[9] = attrs.get('small', 0) + attrs.get('medium', 0) * 0.5 - attrs.get('large', 0)
            text_features[10] = attrs.get('in_air', 0) * 2 - 1
            text_features[11] = attrs.get('wheels', 0) * 2 - 1
            text_features[12] = attrs.get('furry', 0) * 2 - 1
            text_features[13] = attrs.get('legs', 0) * 2 - 1
            text_features[14] = attrs.get('horizontal', 0) * 2 - 1
            text_features[15] = attrs.get('boxy', 0) * 2 - 1
            
            # Add noise
            text_features += np.random.normal(0, noise, text_dim)
            
            # Add weak class-correlated signal in remaining dims (not enough to classify alone)
            text_features[16:32] = np.random.normal(label / 10.0, 0.2, 16)
            
            # Random noise in final dims
            text_features[32:] = np.random.normal(0, 0.1, text_dim - 32)
            
            self.images.append(img_flat)
            self.texts.append(torch.tensor(text_features))
            self.labels.append(label)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.texts[idx], self.labels[idx]


class CorruptedCIFARDataset:
    """Wrapper for CIFAR corruption."""
    
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
            # Gaussian noise
            img = img + self.corruption_level * torch.randn_like(img)
            img = torch.clamp(img, 0, 1)
        elif self.corruption_type == 'occluded_image':
            # Random 16x16 occlusion patch
            mask = torch.ones_like(img)
            # Unflatten to 3x32x32
            mask_3d = mask.view(3, 32, 32)
            x, y = np.random.randint(0, 16, size=2)
            mask_3d[:, x:x+16, y:y+16] = 0
            mask = mask_3d.view(-1)
            img = img * mask
        elif self.corruption_type == 'missing_image':
            img = torch.zeros_like(img)
        elif self.corruption_type == 'text_dropout':
            mask = (torch.rand_like(text) > self.corruption_level).float()
            text = text * mask
        elif self.corruption_type == 'missing_text':
            text = torch.zeros_like(text)
        elif self.corruption_type == 'wrong_text':
            # Add adversarial noise
            text = text + torch.randn_like(text) * self.corruption_level
        elif self.corruption_type == 'color_jitter':
            # Shuffle color channels
            img_3d = img.view(3, 32, 32)
            perm = torch.randperm(3)
            img_3d = img_3d[perm]
            img = img_3d.view(-1)
        
        return img, text, label
