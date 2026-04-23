import torch
import torch.nn as nn
import torch.nn.functional as F


class TCELiteV4(nn.Module):
    """TCE-Lite v4: Abandoning separate encoders for multi-scale fusion.
    
    The insight from v1-v3: early concatenation wins because information
    mixes immediately. v4 embraces this but adds multi-scale processing
    that the baseline can't do.
    
    Key idea: Process at multiple scales (local, mid, global) and fuse,
    creating a hierarchy that can compensate when parts are corrupted.
    """
    def __init__(self, image_dim=64, text_dim=32, n_classes=10):
        super().__init__()
        total_dim = image_dim + text_dim
        
        # Multi-scale processing of concatenated input
        # Local scale: fine-grained features
        self.local_encoder = nn.Sequential(
            nn.Linear(total_dim, 48),
            nn.ReLU(),
            nn.Linear(48, 32),
            nn.ReLU()
        )
        
        # Mid scale: chunk-based processing (simulating manifold patches)
        self.image_chunk_proj = nn.Linear(16, 16)  # each chunk is 16-dim, project to 16
        self.text_chunk_proj = nn.Linear(16, 16)   # each chunk is 16-dim, project to 16
        
        # Cross-chunk attention (the "graph" part)
        self.chunk_attention = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
        
        self.mid_fusion = nn.Sequential(
            nn.Linear(16 * 6, 48),  # 4 image chunks + 2 text chunks
            nn.ReLU(),
            nn.Linear(48, 32),
            nn.ReLU()
        )
        
        # Global scale: full integration
        self.global_fusion = nn.Sequential(
            nn.Linear(32 + 32, 48),  # local + mid
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.ReLU()
        )
        
        # Reliability estimation for each scale
        self.local_reliability = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.mid_reliability = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Final classifier with learned combination
        self.classifier = nn.Linear(48, n_classes)
        
    def forward(self, image, text):
        # Early concatenation (like baseline)
        x = torch.cat([image, text], dim=-1)
        
        # Local scale processing
        local_features = self.local_encoder(x)
        
        # Mid scale: chunk into patches
        batch_size = image.size(0)
        
        # Image chunks (4 chunks of 16)
        img_chunks = image.view(batch_size, 4, 16)  # (B, 4, 16)
        img_chunk_repr = self.image_chunk_proj(img_chunks)  # (B, 4, 16)
        
        # Text chunks (2 chunks of 16)
        txt_chunks = text.view(batch_size, 2, 16)  # (B, 2, 16)
        txt_chunk_repr = self.text_chunk_proj(txt_chunks)  # (B, 2, 16)
        
        # Combine all chunks
        all_chunks = torch.cat([img_chunk_repr, txt_chunk_repr], dim=1)  # (B, 6, 16)
        
        # Cross-chunk attention (allow chunks to attend to each other)
        attended_chunks, _ = self.chunk_attention(all_chunks, all_chunks, all_chunks)
        
        # Flatten and process mid-scale
        mid_flat = attended_chunks.reshape(batch_size, -1)  # (B, 96)
        mid_features = self.mid_fusion(mid_flat)
        
        # Estimate reliability of each scale
        local_weight = self.local_reliability(local_features)
        mid_weight = self.mid_reliability(mid_features)
        
        # Normalize
        total = local_weight + mid_weight + 1e-6
        local_weight = local_weight / total
        mid_weight = mid_weight / total
        
        # Combine scales with learned weights
        combined = torch.cat([
            local_features * local_weight,
            mid_features * mid_weight
        ], dim=-1)
        
        # Global fusion
        global_features = self.global_fusion(combined)
        
        return self.classifier(global_features)


class TCELiteV4WithTrainingWrapper(nn.Module):
    """Wrapper for corruption-aware training."""
    def __init__(self, image_dim=64, text_dim=32, n_classes=10, corruption_prob=0.4):
        super().__init__()
        self.base_model = TCELiteV4(image_dim, text_dim, n_classes)
        self.corruption_prob = corruption_prob
        self.training_mode = True
    
    def forward(self, image, text):
        if self.training and self.training_mode:
            if torch.rand(1).item() < self.corruption_prob:
                corruption = torch.randint(0, 4, (1,)).item()
                if corruption == 0:
                    image = image + 0.3 * torch.randn_like(image)
                elif corruption == 1:
                    text = text + 0.3 * torch.randn_like(text)
                elif corruption == 2:
                    mask = (torch.rand_like(image) > 0.5).float()
                    image = image * mask
                elif corruption == 3:
                    mask = (torch.rand_like(text) > 0.5).float()
                    text = text * mask
        
        return self.base_model(image, text)
    
    def eval(self):
        self.training_mode = False
        return self
    
    def train(self, mode=True):
        self.training_mode = mode
        return self
