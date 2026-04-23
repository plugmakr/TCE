import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoderCIFAR(nn.Module):
    """Encoder for CIFAR images (3072 dims = 3x32x32)."""
    
    def __init__(self, input_dim=3072, hidden_dim=256, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)


class TextEncoderCIFAR(nn.Module):
    """Encoder for CIFAR text descriptions (64-dim)."""
    
    def __init__(self, input_dim=64, hidden_dim=96, output_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)


class CrossModalFusionCIFAR(nn.Module):
    """Fusion with cross-attention and reliability gating."""
    
    def __init__(self, img_dim=128, txt_dim=64, fusion_dim=128):
        super().__init__()
        self.fusion_dim = fusion_dim
        
        # Cross-attention
        self.img_query = nn.Linear(img_dim, fusion_dim)
        self.img_key = nn.Linear(txt_dim, fusion_dim)
        self.img_value = nn.Linear(txt_dim, fusion_dim)
        
        self.txt_query = nn.Linear(txt_dim, fusion_dim)
        self.txt_key = nn.Linear(img_dim, fusion_dim)
        self.txt_value = nn.Linear(img_dim, fusion_dim)
        
        # Self representations
        self.img_self = nn.Linear(img_dim, fusion_dim // 2)
        self.txt_self = nn.Linear(txt_dim, fusion_dim // 2)
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 2 + fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU()
        )
        
        # Reliability estimation
        self.reliability = nn.Sequential(
            nn.Linear(img_dim + txt_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 2),
            nn.Sigmoid()
        )
    
    def forward(self, img_repr, txt_repr):
        batch_size = img_repr.size(0)
        
        # Cross-attention: image attends to text
        img_q = self.img_query(img_repr).unsqueeze(1)
        img_k = self.img_key(txt_repr).unsqueeze(1)
        img_v = self.img_value(txt_repr).unsqueeze(1)
        scores = torch.matmul(img_q, img_k.transpose(-2, -1)) / (self.fusion_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        img_cross = torch.matmul(attn, img_v).squeeze(1)
        
        # Cross-attention: text attends to image
        txt_q = self.txt_query(txt_repr).unsqueeze(1)
        txt_k = self.txt_key(img_repr).unsqueeze(1)
        txt_v = self.txt_value(img_repr).unsqueeze(1)
        scores = torch.matmul(txt_q, txt_k.transpose(-2, -1)) / (self.fusion_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        txt_cross = torch.matmul(attn, txt_v).squeeze(1)
        
        # Self representations
        img_s = self.img_self(img_repr)
        txt_s = self.txt_self(txt_repr)
        
        # Reliability gating
        rel = self.reliability(torch.cat([img_repr, txt_repr], dim=-1))
        img_w, txt_w = rel[:, 0:1], rel[:, 1:2]
        
        # Weighted fusion
        combined = torch.cat([
            img_cross * img_w,
            txt_cross * txt_w,
            img_s,
            txt_s
        ], dim=-1)
        fused = self.fusion_mlp(combined)
        
        return fused


class TCELiteCIFAR(nn.Module):
    """TCE-Lite for CIFAR-10 + text task."""
    
    def __init__(self, image_dim=3072, text_dim=64, n_classes=10):
        super().__init__()
        self.image_encoder = ImageEncoderCIFAR(image_dim, 256, 128)
        self.text_encoder = TextEncoderCIFAR(text_dim, 96, 64)
        self.fusion = CrossModalFusionCIFAR(128, 64, 128)
        self.classifier = nn.Linear(128, n_classes)
    
    def forward(self, image, text):
        img_repr = self.image_encoder(image)
        txt_repr = self.text_encoder(text)
        fused = self.fusion(img_repr, txt_repr)
        return self.classifier(fused)


class BaselineCIFAR(nn.Module):
    """Baseline: early concatenation for CIFAR."""
    
    def __init__(self, image_dim=3072, text_dim=64, n_classes=10):
        super().__init__()
        total_dim = image_dim + text_dim
        
        self.net = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, image, text):
        x = torch.cat([image, text], dim=-1)
        return self.net(x)


class TCELiteCIFARWithCorruption(nn.Module):
    """Wrapper for corruption-aware training."""
    
    def __init__(self, image_dim=3072, text_dim=64, n_classes=10, corruption_prob=0.4):
        super().__init__()
        self.base = TCELiteCIFAR(image_dim, text_dim, n_classes)
        self.corruption_prob = corruption_prob
        self.training_mode = True
    
    def forward(self, image, text):
        if self.training and self.training_mode:
            if torch.rand(1).item() < self.corruption_prob:
                corruption = torch.randint(0, 4, (1,)).item()
                if corruption == 0:
                    # Noise
                    image = image + 0.2 * torch.randn_like(image)
                    image = torch.clamp(image, 0, 1)
                elif corruption == 1:
                    # Text dropout
                    mask = (torch.rand_like(text) > 0.4).float()
                    text = text * mask
                elif corruption == 2:
                    # Occlusion for flattened images
                    mask = torch.ones_like(image)
                    batch_size = image.size(0)
                    for b in range(batch_size):
                        x, y = torch.randint(0, 16, (2,))
                        for c in range(3):
                            for i in range(16):
                                for j in range(16):
                                    idx = c * 1024 + (x + i) * 32 + (y + j)
                                    if idx < 3072:
                                        mask[b, idx] = 0
                    image = image * mask
                elif corruption == 3:
                    # Missing modality
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
