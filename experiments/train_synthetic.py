import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.tce_lite import TCELite
from src.baseline import BaselineMLP
from experiments.synthetic_dataset import SyntheticMultimodalDataset, corrupt_batch


def train_model(model, train_loader, val_loader, device, epochs=20, model_name="Model"):
    """Train a model and return validation accuracy."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    print(f"\nTraining {model_name}...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for img, txt, y in train_loader:
            img, txt, y = img.to(device), txt.to(device), y.to(device)
            
            if isinstance(model, BaselineMLP):
                # Baseline concatenates inputs
                x = torch.cat([img, txt], dim=-1)
                out = model(x)
            else:
                # TCE-Lite processes modalities separately
                out = model(img, txt)
            
            loss = loss_fn(out, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        
        train_acc = correct / total
        print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss:.4f} train_acc={train_acc:.4f}")
    
    # Evaluate on validation set (clean)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img, txt, y in val_loader:
            img, txt, y = img.to(device), txt.to(device), y.to(device)
            
            if isinstance(model, BaselineMLP):
                x = torch.cat([img, txt], dim=-1)
                out = model(x)
            else:
                out = model(img, txt)
            
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    val_acc = correct / total
    print(f"  {model_name} validation accuracy (clean): {val_acc:.4f}")
    return val_acc


def evaluate_corruption(model, val_loader, device, corruption_type, model_name="Model"):
    """Evaluate model under a specific corruption condition."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for img, txt, y in val_loader:
            img, txt, y = img.to(device), txt.to(device), y.to(device)
            
            # Apply corruption
            if corruption_type == "clean":
                pass
            elif corruption_type == "noisy_image":
                img = img + 0.5 * torch.randn_like(img)
            elif corruption_type == "text_dropout":
                mask = (torch.rand_like(txt) > 0.5).float()
                txt = txt * mask
            elif corruption_type == "missing_image":
                img = torch.zeros_like(img)
            elif corruption_type == "missing_text":
                txt = torch.zeros_like(txt)
            
            if isinstance(model, BaselineMLP):
                x = torch.cat([img, txt], dim=-1)
                out = model(x)
            else:
                out = model(img, txt)
            
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    acc = correct / total
    return acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create dataset and split
    full_dataset = SyntheticMultimodalDataset(n_samples=5000)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Train baseline
    baseline = BaselineMLP(input_dim=96, hidden=64).to(device)
    baseline_clean_acc = train_model(baseline, train_loader, val_loader, device, epochs=20, model_name="BaselineMLP")
    
    # Train TCE-Lite
    tce_lite = TCELite(image_dim=64, text_dim=32, n_classes=10).to(device)
    tce_clean_acc = train_model(tce_lite, train_loader, val_loader, device, epochs=20, model_name="TCE-Lite")
    
    # Evaluate under corruption conditions
    corruption_types = ["clean", "noisy_image", "text_dropout", "missing_image", "missing_text"]
    
    print("\n" + "="*70)
    print("CORRUPTION TEST RESULTS")
    print("="*70)
    
    baseline_results = []
    tce_results = []
    
    for corruption in corruption_types:
        baseline_acc = evaluate_corruption(baseline, val_loader, device, corruption, "BaselineMLP")
        tce_acc = evaluate_corruption(tce_lite, val_loader, device, corruption, "TCE-Lite")
        
        baseline_results.append(baseline_acc)
        tce_results.append(tce_acc)
        
        print(f"{corruption:20s}: Baseline={baseline_acc:.4f}, TCE-Lite={tce_acc:.4f}")
    
    # Print formatted table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Model':<12} {'Clean':<10} {'Noisy Img':<12} {'Txt Drop':<12} {'Miss Img':<12} {'Miss Txt':<12}")
    print("-" * 70)
    print(f"{'Baseline':<12} {baseline_results[0]:<10.4f} {baseline_results[1]:<12.4f} {baseline_results[2]:<12.4f} {baseline_results[3]:<12.4f} {baseline_results[4]:<12.4f}")
    print(f"{'TCE-Lite':<12} {tce_results[0]:<10.4f} {tce_results[1]:<12.4f} {tce_results[2]:<12.4f} {tce_results[3]:<12.4f} {tce_results[4]:<12.4f}")
    print("="*70)
    
    # Research interpretation
    print("\nResearch Interpretation:")
    print("If TCE-Lite degrades more gracefully than the baseline under corruption,")
    print("this provides early evidence that structured manifold-style fusion may")
    print("improve robustness. If it does not outperform baseline, that is also")
    print("an important negative result to report honestly.")


if __name__ == "__main__":
    main()
