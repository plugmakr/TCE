import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.tce_lite_v4 import TCELiteV4, TCELiteV4WithTrainingWrapper
from src.baseline import BaselineMLP
from experiments.synthetic_dataset import SyntheticMultimodalDataset


def train_model(model, train_loader, val_loader, device, epochs=30, model_name="Model", use_corruption_aug=False):
    """Train a model and return validation accuracy."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    print(f"\nTraining {model_name}...")
    if use_corruption_aug:
        print("  (using corruption-aware training)")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for img, txt, y in train_loader:
            img, txt, y = img.to(device), txt.to(device), y.to(device)
            
            # Manual corruption augmentation for baseline
            if use_corruption_aug and isinstance(model, BaselineMLP):
                if torch.rand(1).item() < 0.3:
                    corruption = torch.randint(0, 4, (1,)).item()
                    if corruption == 0:
                        img = img + 0.3 * torch.randn_like(img)
                    elif corruption == 1:
                        txt = txt + 0.3 * torch.randn_like(txt)
                    elif corruption == 2:
                        mask = (torch.rand_like(img) > 0.5).float()
                        img = img * mask
                    elif corruption == 3:
                        mask = (torch.rand_like(txt) > 0.5).float()
                        txt = txt * mask
            
            if isinstance(model, BaselineMLP):
                x = torch.cat([img, txt], dim=-1)
                out = model(x)
            else:
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
        if (epoch + 1) % 5 == 0 or epoch == 0:
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
    print("\nTCE-Lite v4: Multi-scale fusion with early concatenation")
    
    # Create dataset and split
    full_dataset = SyntheticMultimodalDataset(n_samples=5000)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Train baseline with corruption augmentation
    print("\n" + "="*70)
    print("BASELINE WITH CORRUPTION-AWARE TRAINING")
    print("="*70)
    baseline = BaselineMLP(input_dim=96, hidden=64).to(device)
    baseline_clean_acc = train_model(baseline, train_loader, val_loader, device, 
                                      epochs=30, model_name="BaselineMLP", 
                                      use_corruption_aug=True)
    
    # Train TCE-Lite v4 with corruption-aware training
    print("\n" + "="*70)
    print("TCE-LITE V4 WITH CORRUPTION-AWARE TRAINING")
    print("="*70)
    print("Strategy: Multi-scale processing (local + mid + global)")
    tce_lite = TCELiteV4WithTrainingWrapper(image_dim=64, text_dim=32, n_classes=10,
                                             corruption_prob=0.4).to(device)
    tce_clean_acc = train_model(tce_lite, train_loader, val_loader, device, 
                                 epochs=30, model_name="TCE-Lite-v4")
    
    # Evaluate under corruption conditions
    corruption_types = ["clean", "noisy_image", "text_dropout", "missing_image", "missing_text"]
    
    print("\n" + "="*70)
    print("CORRUPTION TEST RESULTS")
    print("="*70)
    
    baseline_results = []
    tce_results = []
    
    for corruption in corruption_types:
        baseline_acc = evaluate_corruption(baseline, val_loader, device, corruption, "BaselineMLP")
        tce_acc = evaluate_corruption(tce_lite, val_loader, device, corruption, "TCE-Lite-v4")
        
        baseline_results.append(baseline_acc)
        tce_results.append(tce_acc)
        
        print(f"{corruption:20s}: Baseline={baseline_acc:.4f}, TCE-Lite-v4={tce_acc:.4f}")
    
    # Print formatted table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Model':<16} {'Clean':<10} {'Noisy Img':<12} {'Txt Drop':<12} {'Miss Img':<12} {'Miss Txt':<12}")
    print("-" * 70)
    print(f"{'Baseline':<16} {baseline_results[0]:<10.4f} {baseline_results[1]:<12.4f} {baseline_results[2]:<12.4f} {baseline_results[3]:<12.4f} {baseline_results[4]:<12.4f}")
    print(f"{'TCE-Lite-v4':<16} {tce_results[0]:<10.4f} {tce_results[1]:<12.4f} {tce_results[2]:<12.4f} {tce_results[3]:<12.4f} {tce_results[4]:<12.4f}")
    print("="*70)
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    improvements = []
    for i, corruption in enumerate(corruption_types):
        diff = tce_results[i] - baseline_results[i]
        if corruption != "clean":
            improvements.append((corruption, diff))
    
    wins = sum(1 for _, diff in improvements if diff > 0)
    losses = sum(1 for _, diff in improvements if diff < 0)
    
    print(f"TCE-Lite v4 wins on {wins}/4 corruption conditions")
    print(f"TCE-Lite v4 loses on {losses}/4 corruption conditions")
    
    if wins > losses:
        print("\n*** TCE-Lite v4 shows improvement with multi-scale processing! ***")
        print("The multi-scale approach may provide the robustness benefits.")
    elif wins == losses:
        print("\n*** TCE-Lite v4 is competitive but not clearly better ***")
        print("Need to iterate on the multi-scale fusion approach.")
    else:
        print("\n*** Still negative results ***")
        print("The baseline's simple approach may be optimal for this task.")
        print("Consider: different task, different corruption types, or")
        print("abandoning the manifold hypothesis for this architecture.")
    
    print("\nPer-condition differences:")
    for corruption, diff in improvements:
        status = "WIN" if diff > 0.01 else "LOSS" if diff < -0.01 else "TIE"
        print(f"  {corruption:20s}: {diff:+.4f} ({status})")


if __name__ == "__main__":
    main()
