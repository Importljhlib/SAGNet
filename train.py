import os
import argparse
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR 
from tqdm import tqdm

from SAGNet import SAGNet, Config


# -------------------------------------------------------------
# 1. Dataset
# -------------------------------------------------------------
class PairedDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.input_files = sorted(glob(os.path.join(input_dir, "*.png")))
        self.file_names = [os.path.basename(f) for f in self.input_files]
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        name = self.file_names[idx]
        inp_path = os.path.join(self.input_dir, name)
        tgt_path = os.path.join(self.target_dir, name)

        if not os.path.exists(tgt_path):
            raise FileNotFoundError(f"Target not found for {name}")

        inp = Image.open(inp_path).convert("RGB")
        tgt = Image.open(tgt_path).convert("RGB")

        return self.to_tensor(inp), self.to_tensor(tgt)


# -------------------------------------------------------------
# 2. Utils: PSNR & Graph Plotting
# -------------------------------------------------------------
def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def save_learning_curve(history, save_dir):
    """
    Train Loss, Val Loss, Val PSNR, Learning Rate를 시각화하여 png로 저장
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(20, 5))

    # 1. Train Total Loss
    plt.subplot(1, 4, 1)
    plt.plot(epochs, history['train_loss'], label='Train Total Loss', color='blue')
    plt.title('Train Loss (L1 + Perc)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # 2. Val Loss (Pixel-wise)
    plt.subplot(1, 4, 2)
    plt.plot(epochs, history['val_loss'], label='Val Loss (L1)', color='red')
    plt.title('Validation Loss (L1 only)')
    plt.xlabel('Epochs')
    plt.ylabel('L1 Loss')
    plt.grid(True)
    plt.legend()

    # 3. Val PSNR
    plt.subplot(1, 4, 3)
    plt.plot(epochs, history['val_psnr'], label='Val PSNR', color='green')
    plt.title('Validation PSNR')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    plt.legend()

    # 4. Learning Rate [추가]
    plt.subplot(1, 4, 4)
    plt.plot(epochs, history['lr'], label='Learning Rate', color='purple')
    plt.title('Learning Rate Decay')
    plt.xlabel('Epochs')
    plt.ylabel('LR')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, "learning_curve.png")
    plt.savefig(save_path)
    plt.close()


# -------------------------------------------------------------
# 3. Loss: Perceptual Loss (VGG19)
# -------------------------------------------------------------
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        
        try:
            from torchvision.models import VGG19_Weights
            weights = VGG19_Weights.IMAGENET1K_V1
            vgg = models.vgg19(weights=weights).features
        except ImportError:
            vgg = models.vgg19(pretrained=True).features
        self.loss_network = nn.Sequential(*list(vgg)[:36]).eval()
        
        # 파라미터 Freeze
        for param in self.loss_network.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        
        pred_feat = self.loss_network(pred_norm)
        target_feat = self.loss_network(target_norm)
        
        return F.l1_loss(pred_feat, target_feat)


# -------------------------------------------------------------
# 4. Training Function
# -------------------------------------------------------------
def train(opt):
    task = opt.task
    batch_size = opt.batch
    lr = opt.lr
    num_epochs = opt.epochs
    save_path = opt.save
    
    lambda_perc = 0.05

    # Dataset paths
    train_input = f"/data/jaehyeon/DL/train/{task}/input"
    train_target = f"/data/jaehyeon/DL/train/{task}/target"
    val_input = f"/data/jaehyeon/DL/val/{task}/easy/input"
    val_target = f"/data/jaehyeon/DL/val/{task}/easy/target"

    # 1. Dataset 로드
    train_set = PairedDataset(train_input, train_target)
    val_set = PairedDataset(val_input, val_target)
    
    if len(train_set) == 0:
        raise RuntimeError(f"Train dataset is empty! Check: {train_input}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)

    # 2. Model & Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")
    cfg = Config(sab_placement="none")
    model =  SAGNet(cfg).to(device)

    # 3. Losses & Optimizer
    criterion_pixel = nn.L1Loss()
    criterion_perc = PerceptualLoss().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=5e-6)

    os.makedirs(save_path, exist_ok=True)
    
    history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'lr': []}

    print(f"----- Training Start ({task}) -----")
    print(f"Configs: Epochs={num_epochs}, Batch={batch_size}, LR={lr}")
    print(f"Results will be saved to: {save_path}")

    # ------------------
    # Epoch Loop
    # ------------------
    best_psnr = -1
    best_ckpt_path = os.path.join(save_path, "best_psnr.ckpt")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0

        current_lr = optimizer.param_groups[0]['lr']

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", ncols=120)

        for inp, tgt in pbar:
            inp, tgt = inp.to(device), tgt.to(device)

            optimizer.zero_grad()
            pred = model(inp)

            loss_pix = criterion_pixel(pred, tgt)
            loss_perc = criterion_perc(pred, tgt)
            
            loss = loss_pix + (lambda_perc * loss_perc)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            
            pbar.set_postfix({
                "Total": f"{loss.item():.4f}", 
                "Pix": f"{loss_pix.item():.4f}", 
                "Perc": f"{loss_perc.item():.4f}",
                "LR": f"{current_lr:.1e}"
            })

        avg_train_loss = total_train_loss / len(train_loader)

        # ------------------
        # Validation
        # ------------------
        model.eval()
        total_val_loss = 0
        total_val_psnr = 0

        if len(val_loader) > 0:
            with torch.no_grad():
                for inp, tgt in val_loader:
                    inp, tgt = inp.to(device), tgt.to(device)
                    pred = model(inp)

                    total_val_loss += criterion_pixel(pred, tgt).item()
                    total_val_psnr += psnr(pred, tgt).item()

            avg_val_loss = total_val_loss / len(val_loader)
            avg_val_psnr = total_val_psnr / len(val_loader)
        else:
            avg_val_loss = 0
            avg_val_psnr = 0

        print(f"[Epoch {epoch}] Train: {avg_train_loss:.4f} | Val L1: {avg_val_loss:.4f} | PSNR: {avg_val_psnr:.2f} | LR: {current_lr:.2e}")

        scheduler.step()

        # ------------------
        # Save & Plot
        # ------------------
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_psnr'].append(avg_val_psnr)
        history['lr'].append(current_lr)

        save_learning_curve(history, save_path)

        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            try:
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"[✔] Best PSNR updated! Saved checkpoint: {best_ckpt_path}")
            except Exception as e:
                print(f"Error saving best checkpoint: {e}")
        else:
            print("[ ] No improvement. Checkpoint not saved.")


    final_ckpt = os.path.join(save_path, "model_final.ckpt")
    torch.save(model.state_dict(), final_ckpt)
    print(f"[OK] Training Finished. Final model: {final_ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Task name (folder name)")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save", type=str, default="./checkpoints") 
    
    args = parser.parse_args()

    train(args)
