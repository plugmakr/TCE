import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoderMNIST(nn.Module):
    """Encoder for flattened MNIST images (784 -> hidden -> output)."""
    
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)


class TextEncoderMNIST(nn.Module):
    """Encoder for text attributes (32-dim)."""
    
    def __init__(self, input_dim=32, hidden_dim=48, output_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)


class CrossModalFusion(nn.Module):
    """Fusion with cross-attention between modalities."""
    
    def __init__(self, img_dim=64, txt_dim=32, fusion_dim=64):
        super().__init__()
        self.fusion_dim = fusion_dim
        
        # Cross-attention projections
        self.img_query = nn.Linear(img_dim, fusion_dim)
        self.img_key = nn.Linear(txt_dim, fusion_dim)
        self.img_value = nn.Linear(txt_dim, fusion_dim)
        
        self.txt_query = nn.Linear(txt_dim, fusion_dim)
        self.txt_key = nn.Linear(img_dim, fusion_dim)
        self.txt_value = nn.Linear(img_dim, fusion_dim)
        
        # Self representations
        self.img_self = nn.Linear(img_dim, fusion_dim // 2)
        self.txt_self = nn.Linear(txt_dim, fusion_dim // 2)
        
        # Fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 2 + fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU()
        )
        
        # Reliability gates
        self.reliability = nn.Sequential(
            nn.Linear(img_dim + txt_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 2),
            nn.Sigmoid()
        )
    
    def forward(self, img_repr, txt_repr):
        batch_size = img_repr.size(0)
        
        # Cross-attention
        img_q = self.img_query(img_repr).unsqueeze(1)
        img_k = self.img_key(txt_repr).unsqueeze(1)
        img_v = self.img_value(txt_repr).unsqueeze(1)
        img_cross = torch.matmul(F.softmax(torch.matmul(img_q, img_k.transpose(-2, -1)) / (self.fusion_dim ** 0.5), dim=-1), img_v).squeeze(1)
        
        txt_q = self.txt_query(txt_repr).unsqueeze(1)
        txt_k = self.txt_key(img_repr).unsqueeze(1)
        txt_v = self.txt_value(img_repr).unsqueeze(1)
        txt_cross = torch.matmul(F.softmax(torch.matmul(txt_q, txt_k.transpose(-2, -1)) / (self.fusion_dim ** 0.5), dim=-1), txt_v).squeeze(1)
        
        # Self representations
        img_s = self.img_self(img_repr)
        txt_s = self.txt_self(txt_repr)
        
        # Reliability weights
        rel_weights = self.reliability(torch.cat([img_repr, txt_repr], dim=-1))
        img_weight, txt_weight = rel_weights[:, 0:1], rel_weights[:, 1:2]
        
        # Weighted fusion
        combined = torch.cat([img_cross * img_weight, txt_cross * txt_weight, 
                              img_s, txt_s], dim=-1)
        fused = self.fusion_mlp(combined)
        
        return fused, img_weight.mean().item(), txt_weight.mean().item()


class TCELiteMNIST(nn.Module):
    """TCE-Lite for MNIST + text task."""
    
    def __init__(self, image_dim=784, text_dim=32, n_classes=10):
        super().__init__()
        self.image_encoder = ImageEncoderMNIST(image_dim, 128, 64)
        self.text_encoder = TextEncoderMNIST(text_dim, 48, 32)
        self.fusion = CrossModalFusion(64, 32, 64)
        self.classifier = nn.Linear(64, n_classes)
    
    def forward(self, image, text):
        img_repr = self.image_encoder(image)
        txt_repr = self.text_encoder(text)
        fused, img_rel, txt_rel = self.fusion(img_repr, txt_repr)
        return self.classifier(fused)


class BaselineMNIST(nn.Module):
    """Baseline for MNIST + text (early concatenation)."""
    
    def __init__(self, image_dim=784, text_dim=32, n_classes=10):
        super().__init__()
        total_dim = image_dim + text_dim
        
        self.net = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, image, text):
        x = torch.cat([image, text], dim=-1)
        return self.net(x)


class TCELiteMNISTWithCorruption(nn.Module):
    """Wrapper for corruption-aware training."""
    
    def __init__(self, image_dim=784, text_dim=32, n_classes=10, corruption_prob=0.3):
        super().__init__()
        self.base = TCELiteMNIST(image_dim, text_dim, n_classes)
        self.corruption_prob = corruption_prob
        self.training_mode = True
    
    def forward(self, image, text):
        if self.training and self.training_mode:
            if torch.rand(1).item() < self.corruption_prob:
                corruption = torch.randint(0, 4, (1,)).item()
                if corruption == 0:
                    image = image + 0.2 * torch.randn_like(image)
                    image = torch.clamp(image, 0, 1)
                elif corruption == 1:
                    mask = (torch.rand_like(text) > 0.3).float()
                    text = text * mask
                elif corruption == 2:
                    mask = torch.ones_like(image)
                    x, y = torch.randint(0, 14, (2,))
                    mask[x:x+14, y:y+14] = 0
                    image = image * mask
                elif corruption == 3:
                    if torch.rand(1).item() < 0.5:
                        image = torch.zeros_like(image)
                    else:
                        text = torch.zeros_like(text)
        return self.base(image, text)
    
    def eval(self):
        self.training_mode = False
        return self
    
    def train(self, mode=True):
        self.training_mode = mode
        return self
