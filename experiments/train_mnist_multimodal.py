import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from experiments.realistic_dataset import MNISTTextDataset, CorruptedMNISTTextDataset
from src.tce_lite_mnist import TCELiteMNIST, BaselineMNIST, TCELiteMNISTWithCorruption


def train_model(model, train_loader, val_loader, device, epochs=20, model_name="Model", 
                use_corruption_aug=False):
    """Train a model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    print(f"\nTraining {model_name}...")
    if use_corruption_aug:
        print("  (with corruption augmentation)")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for img, txt, y in train_loader:
            img, txt, y = img.to(device), txt.to(device), y.to(device)
            
            # Manual corruption augmentation for baseline
            if use_corruption_aug and isinstance(model, BaselineMNIST):
                if torch.rand(1).item() < 0.3:
                    corruption = torch.randint(0, 3, (1,)).item()
                    if corruption == 0:
                        img = img + 0.2 * torch.randn_like(img)
                        img = torch.clamp(img, 0, 1)
                    elif corruption == 1:
                        mask = (torch.rand_like(txt) > 0.3).float()
                        txt = txt * mask
                    elif corruption == 2:
                        mask = torch.ones_like(img)
                        x, y_pos = torch.randint(0, 14, (2,))
                        mask[x:x+14, y_pos:y_pos+14] = 0
                        img = img * mask
            
            out = model(img, txt)
            loss = loss_fn(out, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss:.4f} acc={correct/total:.4f}")
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img, txt, y in val_loader:
            img, txt, y = img.to(device), txt.to(device), y.to(device)
            out = model(img, txt)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    val_acc = correct / total
    print(f"  {model_name} validation accuracy (clean): {val_acc:.4f}")
    return val_acc


def evaluate_on_corruption(base_dataset, model, device, corruption_type, level=0.5):
    """Evaluate on corrupted data."""
    model.eval()
    
    # Create corrupted dataset
    corrupted = CorruptedMNISTTextDataset(base_dataset, corruption_type, level)
    loader = DataLoader(corrupted, batch_size=128, shuffle=False)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for img, txt, y in loader:
            img, txt, y = img.to(device), txt.to(device), y.to(device)
            out = model(img, txt)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    return correct / total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("\n" + "="*70)
    print("MNIST + TEXT MULTIMODAL TASK")
    print("="*70)
    print("Task: Classify digits based on image AND text attributes")
    print("Neither modality alone can determine the class!")
    
    # Load datasets
    print("\nLoading MNIST dataset...")
    full_train = MNISTTextDataset(root='./data', train=True, text_dim=32)
    test_dataset = MNISTTextDataset(root='./data', train=False, text_dim=32)
    
    # Split train into train/val
    train_size = int(0.8 * len(full_train))
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Train baseline with corruption augmentation
    print("\n" + "="*70)
    print("BASELINE MLP")
    print("="*70)
    baseline = BaselineMNIST(image_dim=784, text_dim=32, n_classes=10).to(device)
    baseline_clean = train_model(baseline, train_loader, val_loader, device, 
                                  epochs=20, model_name="Baseline", use_corruption_aug=True)
    
    # Train TCE-Lite
    print("\n" + "="*70)
    print("TCE-LITE MNIST")
    print("="*70)
    tce = TCELiteMNISTWithCorruption(image_dim=784, text_dim=32, n_classes=10, 
                                      corruption_prob=0.4).to(device)
    tce_clean = train_model(tce, train_loader, val_loader, device, 
                            epochs=20, model_name="TCE-Lite")
    
    # Test on various corruptions
    print("\n" + "="*70)
    print("CORRUPTION TESTS")
    print("="*70)
    
    corruptions = [
        ('clean', 0),
        ('noisy_image', 0.3),
        ('missing_image', 0),
        ('text_dropout', 0.5),
        ('missing_text', 0),
        ('wrong_text', 0.5),
        ('occluded_image', 0),
    ]
    
    baseline_results = []
    tce_results = []
    
    for corruption, level in corruptions:
        b_acc = evaluate_on_corruption(test_dataset, baseline, device, corruption, level)
        t_acc = evaluate_on_corruption(test_dataset, tce, device, corruption, level)
        baseline_results.append(b_acc)
        tce_results.append(t_acc)
        
        print(f"{corruption:20s}: Baseline={b_acc:.4f}, TCE-Lite={t_acc:.4f}, diff={t_acc-b_acc:+.4f}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"{'Condition':<20} {'Baseline':<12} {'TCE-Lite':<12} {'Diff':<10}")
    print("-" * 70)
    for i, (corr, _) in enumerate(corruptions):
        diff = tce_results[i] - baseline_results[i]
        status = "WIN" if diff > 0.01 else "LOSS" if diff < -0.01 else "TIE"
        print(f"{corr:<20} {baseline_results[i]:<12.4f} {tce_results[i]:<12.4f} {diff:+.4f} {status}")
    
    # Count wins
    wins = sum(1 for i in range(1, len(corruptions)) if tce_results[i] > baseline_results[i] + 0.01)
    losses = sum(1 for i in range(1, len(corruptions)) if tce_results[i] < baseline_results[i] - 0.01)
    
    print("\n" + "="*70)
    if wins > losses:
        print(f"*** TCE-Lite WINS: {wins} vs {losses} ***")
        print("Multi-modal fusion with separate encoders shows benefit!")
    elif wins == losses:
        print(f"TIE: TCE-Lite wins on {wins}, loses on {losses}")
        print("Performance is comparable between architectures")
    else:
        print(f"BASELINE WINS: {wins} vs {losses}")
        print("Early concatenation remains optimal for this task")
    print("="*70)


if __name__ == "__main__":
    main()
