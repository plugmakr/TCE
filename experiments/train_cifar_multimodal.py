import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from experiments.cifar_dataset import CIFARTextDataset, CorruptedCIFARDataset
from src.tce_lite_cifar import TCELiteCIFAR, BaselineCIFAR, TCELiteCIFARWithCorruption


def train_model(model, train_loader, val_loader, device, epochs=25, model_name="Model", use_corruption_aug=False):
    """Train a model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    print(f"\nTraining {model_name}...")
    if use_corruption_aug:
        print("  (with corruption augmentation)")
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for img, txt, y in train_loader:
            img, txt, y = img.to(device), txt.to(device), y.to(device)
            
            # Manual corruption augmentation for baseline
            if use_corruption_aug and isinstance(model, BaselineCIFAR):
                if torch.rand(1).item() < 0.3:
                    corruption = torch.randint(0, 3, (1,)).item()
                    if corruption == 0:
                        img = img + 0.2 * torch.randn_like(img)
                        img = torch.clamp(img, 0, 1)
                    elif corruption == 1:
                        mask = (torch.rand_like(txt) > 0.3).float()
                        txt = txt * mask
                    elif corruption == 2:
                        # Occlusion for flattened image (batch_size, 3072)
                        mask = torch.ones_like(img)
                        batch_size = img.size(0)
                        for b in range(batch_size):
                            x, y_pos = torch.randint(0, 16, (2,))
                            for c in range(3):
                                for i in range(16):
                                    for j in range(16):
                                        idx = c * 1024 + (x + i) * 32 + (y_pos + j)
                                        if idx < 3072:
                                            mask[b, idx] = 0
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
        
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for img, txt, y in val_loader:
                img, txt, y = img.to(device), txt.to(device), y.to(device)
                out = model(img, txt)
                preds = out.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        
        val_acc = val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss:.4f} train={train_acc:.4f} val={val_acc:.4f}")
    
    print(f"  {model_name} best validation accuracy: {best_val_acc:.4f}")
    return best_val_acc


def evaluate_on_corruption(base_dataset, model, device, corruption_type, level=0.5):
    """Evaluate on corrupted data."""
    model.eval()
    
    corrupted = CorruptedCIFARDataset(base_dataset, corruption_type, level)
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
    print("CIFAR-10 + TEXT MULTIMODAL TASK")
    print("="*70)
    print("Task: Classify CIFAR images with text attribute descriptions")
    print("Significantly harder than MNIST - 32x32 color images, 10 classes")
    
    # Load datasets
    print("\nLoading CIFAR-10 dataset...")
    full_train = CIFARTextDataset(root='./data', train=True, text_dim=64)
    test_dataset = CIFARTextDataset(root='./data', train=False, text_dim=64)
    
    # Split train
    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # Train baseline
    print("\n" + "="*70)
    print("BASELINE CIFAR")
    print("="*70)
    baseline = BaselineCIFAR(image_dim=3072, text_dim=64, n_classes=10).to(device)
    baseline_clean = train_model(baseline, train_loader, val_loader, device, 
                                  epochs=25, model_name="Baseline", use_corruption_aug=True)
    
    # Train TCE-Lite
    print("\n" + "="*70)
    print("TCE-LITE CIFAR")
    print("="*70)
    tce = TCELiteCIFARWithCorruption(image_dim=3072, text_dim=64, n_classes=10, 
                                      corruption_prob=0.4).to(device)
    tce_clean = train_model(tce, train_loader, val_loader, device, 
                            epochs=25, model_name="TCE-Lite")
    
    # Corruption tests
    print("\n" + "="*70)
    print("CORRUPTION TESTS")
    print("="*70)
    
    corruptions = [
        ('clean', 0),
        ('noisy_image', 0.3),
        ('occluded_image', 0),
        ('missing_image', 0),
        ('text_dropout', 0.5),
        ('missing_text', 0),
        ('wrong_text', 0.5),
        ('color_jitter', 0),
    ]
    
    baseline_results = []
    tce_results = []
    
    for corruption, level in corruptions:
        b_acc = evaluate_on_corruption(test_dataset, baseline, device, corruption, level)
        t_acc = evaluate_on_corruption(test_dataset, tce, device, corruption, level)
        baseline_results.append(b_acc)
        tce_results.append(t_acc)
        
        diff = t_acc - b_acc
        status = "WIN" if diff > 0.01 else "LOSS" if diff < -0.01 else "TIE"
        print(f"{corruption:20s}: Baseline={b_acc:.4f}, TCE-Lite={t_acc:.4f}, diff={diff:+.4f} [{status}]")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Condition':<18} {'Baseline':<10} {'TCE-Lite':<10} {'Diff':<8} {'Status':<6}")
    print("-" * 70)
    for i, (corr, _) in enumerate(corruptions):
        diff = tce_results[i] - baseline_results[i]
        status = "WIN" if diff > 0.01 else "LOSS" if diff < -0.01 else "TIE"
        print(f"{corr:<18} {baseline_results[i]:<10.4f} {tce_results[i]:<10.4f} {diff:<+8.4f} {status:<6}")
    
    # Wins
    wins = sum(1 for i in range(1, len(corruptions)) if tce_results[i] > baseline_results[i] + 0.01)
    losses = sum(1 for i in range(1, len(corruptions)) if tce_results[i] < baseline_results[i] - 0.01)
    
    print("\n" + "="*70)
    if wins > losses:
        print(f"*** TCE-LITE WINS: {wins} vs {losses} on harder CIFAR task! ***")
        print("Manifold architecture shows robustness benefits at scale!")
    elif wins == losses:
        print(f"TIE: TCE-Lite wins on {wins}, loses on {losses}")
        print("Performance is comparable - need harder task or better architecture")
    else:
        print(f"BASELINE WINS: {wins} vs {losses}")
        print("Early concatenation remains stronger on CIFAR")
        print("May need deeper architecture or different fusion approach")
    print("="*70)


if __name__ == "__main__":
    main()
