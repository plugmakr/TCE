import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.tce_lite import SimpleTCE
from experiments.synthetic_dataset import SyntheticMultimodalDataset, corrupt_batch


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SyntheticMultimodalDataset(n_samples=5000)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = SimpleTCE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        total_loss = 0
        correct = 0
        total = 0

        for img, txt, y in loader:
            img, txt, y = img.to(device), txt.to(device), y.to(device)

            # combine modalities (baseline style for now)
            x = torch.cat([img, txt], dim=-1)

            # corruption test during training (light)
            x_img, x_txt = corrupt_batch(img, txt, image_noise=0.02, text_dropout=0.05)
            x = torch.cat([x_img, x_txt], dim=-1)

            out = model(x)
            loss = loss_fn(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}: loss={total_loss:.4f} acc={acc:.4f}")


if __name__ == "__main__":
    train()
