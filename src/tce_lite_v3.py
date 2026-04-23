import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalEncoder(nn.Module):
    """Encoder with early cross-modal connections.
    
    Unlike the isolated encoders in v2, this allows information to flow
    between modalities at each layer through residual cross-connections.
    This is closer to how the baseline works (early mixing) while still
    maintaining separate manifold structures.
    """
    def __init__(self, image_dim=64, text_dim=32, hidden_dim=48, output_dim=32):
        super().__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        
        # Initial projections
        self.img_proj = nn.Linear(image_dim, hidden_dim)
        self.txt_proj = nn.Linear(text_dim, hidden_dim)
        
        # Cross-modal connections (the key fix)
        self.img_to_txt = nn.Linear(hidden_dim, hidden_dim)
        self.txt_to_img = nn.Linear(hidden_dim, hidden_dim)
        
        # Second layer
        self.img_layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.txt_layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Output projections
        self.img_out = nn.Linear(hidden_dim, output_dim)
        self.txt_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, image, text):
        # Initial projections
        img_h = F.relu(self.img_proj(image))
        txt_h = F.relu(self.txt_proj(text))
        
        # Cross-modal residual connections (CRITICAL FIX)
        # Information flows both ways at the hidden layer
        img_cross = self.img_to_txt(txt_h)
        txt_cross = self.txt_to_img(img_h)
        
        # Add residual cross-connections
        img_h = img_h + 0.3 * img_cross  # scaled residual
        txt_h = txt_h + 0.3 * txt_cross
        
        # Second layer
        img_h = self.img_layer2(img_h)
        txt_h = self.txt_layer2(txt_h)
        
        # Output representations
        img_repr = F.relu(self.img_out(img_h))
        txt_repr = F.relu(self.txt_out(txt_h))
        
        return img_repr, txt_repr


class AdaptiveFusion(nn.Module):
    """Adaptive fusion that learns to combine based on modality reliability.
    
    This fusion mechanism estimates how much to trust each modality
    and combines them accordingly. The key insight: it should learn
    to weight modalities based on their input characteristics.
    """
    def __init__(self, img_dim=32, txt_dim=32, fusion_dim=48, n_classes=10):
        super().__init__()
        
        # Reliability estimators (look at input statistics)
        self.img_reliability = nn.Sequential(
            nn.Linear(img_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.txt_reliability = nn.Sequential(
            nn.Linear(txt_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Project to common space
        self.img_proj = nn.Linear(img_dim, fusion_dim)
        self.txt_proj = nn.Linear(txt_dim, fusion_dim)
        
        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU()
        )
        
        # Classifier
        self.classifier = nn.Linear(fusion_dim, n_classes)
        
    def forward(self, img_repr, txt_repr):
        # Estimate reliability of each modality
        img_weight = self.img_reliability(img_repr)
        txt_weight = self.txt_reliability(txt_repr)
        
        # Normalize weights
        total_weight = img_weight + txt_weight + 1e-6
        img_weight = img_weight / total_weight
        txt_weight = txt_weight / total_weight
        
        # Weight and project
        img_proj = self.img_proj(img_repr) * img_weight
        txt_proj = self.txt_proj(txt_repr) * txt_weight
        
        # Combine
        combined = torch.cat([img_proj, txt_proj], dim=-1)
        fused = self.fusion(combined)
        
        return self.classifier(fused), img_weight, txt_weight


class TCELiteV3(nn.Module):
    """TCE-Lite v3: With early cross-modal connections and adaptive fusion.
    
    Key improvements over v2:
    1. Cross-modal residual connections in encoders (information flows early)
    2. Adaptive fusion that learns reliability weighting
    3. Designed to handle missing/noisy modalities better
    """
    def __init__(self, image_dim=64, text_dim=32, n_classes=10):
        super().__init__()
        self.encoder = CrossModalEncoder(image_dim, text_dim, 48, 32)
        self.fusion = AdaptiveFusion(32, 32, 48, n_classes)
    
    def forward(self, image, text):
        # Encode with cross-modal connections
        img_repr, txt_repr = self.encoder(image, text)
        
        # Adaptive fusion and classification
        output, img_weight, txt_weight = self.fusion(img_repr, txt_repr)
        
        return output


class TCELiteV3WithCorruptionTraining(nn.Module):
    """TCE-Lite v3 with corruption-aware training wrapper.
    
    This version is designed to be trained with random corruption
    so it learns robust representations from the start.
    """
    def __init__(self, image_dim=64, text_dim=32, n_classes=10, 
                 corruption_prob=0.3, noise_level=0.3):
        super().__init__()
        self.base_model = TCELiteV3(image_dim, text_dim, n_classes)
        self.corruption_prob = corruption_prob
        self.noise_level = noise_level
        self.training_mode = True
    
    def forward(self, image, text):
        # During training, randomly corrupt inputs
        if self.training and self.training_mode:
            if torch.rand(1).item() < self.corruption_prob:
                # Random corruption type
                corruption = torch.randint(0, 4, (1,)).item()
                if corruption == 0:
                    # Add noise to image
                    image = image + self.noise_level * torch.randn_like(image)
                elif corruption == 1:
                    # Add noise to text
                    text = text + self.noise_level * torch.randn_like(text)
                elif corruption == 2:
                    # Dropout on image
                    mask = (torch.rand_like(image) > 0.5).float()
                    image = image * mask
                elif corruption == 3:
                    # Dropout on text
                    mask = (torch.rand_like(text) > 0.5).float()
                    text = text * mask
        
        return self.base_model(image, text)
    
    def eval(self):
        self.training_mode = False
        return self
    
    def train(self, mode=True):
        self.training_mode = mode
        return self
