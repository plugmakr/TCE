import torch
import torch.nn as nn


class ImageManifoldEncoder(nn.Module):
    """Encodes image-like vectors into a manifold representation.
    
    In TCE, this represents the spatial / HWCM manifold side of the architecture.
    For this lite version, we use a simple MLP. In a full implementation,
    this would be replaced with a Hierarchical Waffle Cubical Manifold (HWCM)
    graph structure.
    """
    def __init__(self, input_dim=64, hidden_dim=32, output_dim=16):
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
    """
    def __init__(self, input_dim=32, hidden_dim=16, output_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)


class SIPFusionLayer(nn.Module):
    """Shared Integration Point (SIP) fusion layer.
    
    In TCE, the SIP is where multiple manifolds connect through smooth
    transition mappings. This layer combines manifold-level summaries
    from different encoders into a shared representation space.
    """
    def __init__(self, image_dim=16, text_dim=8, fusion_dim=16):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(image_dim + text_dim, fusion_dim),
            nn.ReLU()
        )
    
    def forward(self, image_repr, text_repr):
        # Concatenate manifold representations
        combined = torch.cat([image_repr, text_repr], dim=-1)
        return self.fusion(combined)


class ClassifierHead(nn.Module):
    """Final classification head.
    
    Takes the fused representation from the SIP and produces class predictions.
    """
    def __init__(self, input_dim=16, n_classes=10):
        super().__init__()
        self.classifier = nn.Linear(input_dim, n_classes)
    
    def forward(self, x):
        return self.classifier(x)


class TCELite(nn.Module):
    """TCE-Lite: Minimal topology-inspired multimodal architecture.
    
    This prototype demonstrates the core TCE concept:
    - Separate encoders for different modalities (manifolds)
    - A shared integration point (SIP) for fusion
    - This structure may provide more graceful degradation under corruption
      compared to simple concatenation baselines.
    """
    def __init__(self, image_dim=64, text_dim=32, n_classes=10):
        super().__init__()
        self.image_encoder = ImageManifoldEncoder(image_dim, 32, 16)
        self.text_encoder = TextManifoldEncoder(text_dim, 16, 8)
        self.sip_fusion = SIPFusionLayer(16, 8, 16)
        self.classifier = ClassifierHead(16, n_classes)
    
    def forward(self, image, text):
        # Encode each modality into its manifold representation
        image_repr = self.image_encoder(image)
        text_repr = self.text_encoder(text)
        
        # Fuse at the Shared Integration Point
        fused = self.sip_fusion(image_repr, text_repr)
        
        # Classify
        return self.classifier(fused)
