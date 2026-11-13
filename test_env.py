import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import hydra
from omegaconf import DictConfig

class KvasirSegDataset(Dataset):
    def __init__(self, root_dir, image_size=(256, 256)):
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "images", "*.jpg")))
        self.mask_paths = sorted(glob.glob(os.path.join(root_dir, "masks", "*.jpg")))
        self.img_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor()
        ])
        self.mask_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()
        return img, mask

class TinyUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, num_classes, 1)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def compute_dice_iou(pred, target, eps=1e-7):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    iou = (intersection + eps) / (union - intersection + eps)
    return dice.mean().item(), iou.mean().item()

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print("=" * 50)
    print("KIỂM TRA CẤU HÌNH VÀ MÔI TRƯỜNG DỰ ÁN")
    print("=" * 50)
    print("1. Hydra Configuration:")
    print(f"   Project: {cfg.project.name}")
    print(f"   Model: {cfg.model.name}")
    print(f"   Dataset root: {cfg.dataset.root_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train.device == "cuda" else "cpu")
    print(f"2. Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode'})")

    print("3. Kiểm tra Dataloader Kvasir-SEG...")
    dataset = KvasirSegDataset(cfg.dataset.root_dir, tuple(cfg.dataset.image_size))
    print(f"   Số lượng mẫu tìm thấy: {len(dataset)} ảnh & mask.")
    assert len(dataset) > 0, "Không tìm thấy ảnh trong dataset!"

    loader = DataLoader(dataset, batch_size=cfg.dataset.batch_size, shuffle=True)
    images, masks = next(iter(loader))
    images, masks = images.to(device), masks.to(device)
    print(f"   Shape batch ảnh: {images.shape}")
    print(f"   Shape batch mask: {masks.shape}")

    print("4. Chạy thử mô hình (Sanity forward pass)...")
    model = TinyUNet().to(device)
    outputs = model(images)
    print(f"   Output logits shape: {outputs.shape}")

    dice, iou = compute_dice_iou(outputs, masks)
    print(f"5. Tính metric mẫu:")
    print(f"   Dice Score: {dice:.4f}")
    print(f"   IoU Score:  {iou:.4f}")

    print("=" * 50)
    print(">>> MÔI TRƯỜNG ĐÃ SẴN SÀNG!")
    print("=" * 50)

if __name__ == "__main__":
    main()