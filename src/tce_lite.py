import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageManifoldEncoder(nn.Module):
    """Encodes image-like vectors into a manifold representation.
    
    In TCE, this represents the spatial / HWCM manifold side of the architecture.
    Scaled up to match baseline parameter count for fair comparison.
    """
    def __init__(self, input_dim=64, hidden_dim=56, output_dim=40):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)


class TextManifoldEncoder(nn.Module):
    """Encodes text-like symbolic vectors into a manifold representation.
    
    In TCE, this represents the symbolic manifold side with different
    inductive biases than the spatial manifold.
    Scaled up to match baseline parameter count for fair comparison.
    """
    def __init__(self, input_dim=32, hidden_dim=28, output_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)


class GraphFusionSIP(nn.Module):
    """Graph-based Shared Integration Point (SIP) fusion layer.
    
    In TCE, the SIP is where multiple manifolds connect through smooth
    transition mappings. This implementation uses a graph attention approach
    where image and text nodes exchange information through learnable edges.
    
    This creates a true fusion where each modality can attend to the other,
    rather than simple concatenation.
    """
    def __init__(self, image_dim=40, text_dim=20, fusion_dim=48, num_heads=4):
        super().__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.fusion_dim = fusion_dim
        
        # Cross-modal attention: image attends to text
        self.img_query = nn.Linear(image_dim, fusion_dim)
        self.img_key = nn.Linear(text_dim, fusion_dim)
        self.img_value = nn.Linear(text_dim, fusion_dim)
        
        # Cross-modal attention: text attends to image  
        self.txt_query = nn.Linear(text_dim, fusion_dim)
        self.txt_key = nn.Linear(image_dim, fusion_dim)
        self.txt_value = nn.Linear(image_dim, fusion_dim)
        
        # Self-attention for each modality
        self.img_self_query = nn.Linear(image_dim, fusion_dim // 2)
        self.img_self_key = nn.Linear(image_dim, fusion_dim // 2)
        self.img_self_value = nn.Linear(image_dim, fusion_dim // 2)
        
        self.txt_self_query = nn.Linear(text_dim, fusion_dim // 2)
        self.txt_self_key = nn.Linear(text_dim, fusion_dim // 2)
        self.txt_self_value = nn.Linear(text_dim, fusion_dim // 2)
        
        # Fusion MLP combining all representations
        total_dim = fusion_dim * 2 + (fusion_dim // 2) * 2
        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU()
        )
        
        # Learnable gating for modality importance
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )
    
    def attention(self, query, key, value):
        """Scaled dot-product attention."""
        scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, value)
    
    def forward(self, image_repr, text_repr):
        batch_size = image_repr.size(0)
        
        # Cross-modal: image attends to text
        img_q = self.img_query(image_repr).unsqueeze(1)  # (B, 1, fusion_dim)
        img_k = self.img_key(text_repr).unsqueeze(1)     # (B, 1, fusion_dim)
        img_v = self.img_value(text_repr).unsqueeze(1)   # (B, 1, fusion_dim)
        img_cross = self.attention(img_q, img_k, img_v).squeeze(1)  # (B, fusion_dim)
        
        # Cross-modal: text attends to image
        txt_q = self.txt_query(text_repr).unsqueeze(1)
        txt_k = self.txt_key(image_repr).unsqueeze(1)
        txt_v = self.txt_value(image_repr).unsqueeze(1)
        txt_cross = self.attention(txt_q, txt_k, txt_v).squeeze(1)
        
        # Self-attention for image
        img_sq = self.img_self_query(image_repr).unsqueeze(1)
        img_sk = self.img_self_key(image_repr).unsqueeze(1)
        img_sv = self.img_self_value(image_repr).unsqueeze(1)
        img_self = self.attention(img_sq, img_sk, img_sv).squeeze(1)
        
        # Self-attention for text
        txt_sq = self.txt_self_query(text_repr).unsqueeze(1)
        txt_sk = self.txt_self_key(text_repr).unsqueeze(1)
        txt_sv = self.txt_self_value(text_repr).unsqueeze(1)
        txt_self = self.attention(txt_sq, txt_sk, txt_sv).squeeze(1)
        
        # Combine all representations
        combined = torch.cat([img_cross, txt_cross, img_self, txt_self], dim=-1)
        fused = self.fusion_mlp(combined)
        
        # Apply learned gating
        gate_input = torch.cat([img_cross, txt_cross], dim=-1)
        gate = self.gate(gate_input)
        
        return fused * gate


class ClassifierHead(nn.Module):
    """Final classification head.
    
    Takes the fused representation from the SIP and produces class predictions.
    """
    def __init__(self, input_dim=48, n_classes=10):
        super().__init__()
        self.classifier = nn.Linear(input_dim, n_classes)
    
    def forward(self, x):
        return self.classifier(x)


class TCELite(nn.Module):
    """TCE-Lite: Topology-inspired multimodal architecture with graph fusion.
    
    This implementation:
    - Separate encoders for different modalities (manifolds)
    - Graph-based SIP fusion with cross-modal attention
    - Similar parameter count to baseline for fair comparison
    - Structure may provide more graceful degradation under corruption
    """
    def __init__(self, image_dim=64, text_dim=32, n_classes=10):
        super().__init__()
        self.image_encoder = ImageManifoldEncoder(image_dim, 56, 40)
        self.text_encoder = TextManifoldEncoder(text_dim, 28, 20)
        self.sip_fusion = GraphFusionSIP(40, 20, 48)
        self.classifier = ClassifierHead(48, n_classes)
    
    def forward(self, image, text):
        # Encode each modality into its manifold representation
        image_repr = self.image_encoder(image)
        text_repr = self.text_encoder(text)
        
        # Fuse at the Shared Integration Point with graph-based fusion
        fused = self.sip_fusion(image_repr, text_repr)
        
        # Classify
        return self.classifier(fused)
