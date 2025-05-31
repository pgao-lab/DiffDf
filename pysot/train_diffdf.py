import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import time
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from diffusers import UNet2DModel, DDPMScheduler
from torchvision import transforms, models
from tqdm import tqdm
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser(description="Train a DDPM model for image denoising.")
    parser.add_argument("--image_folder", type=str, default='',
                        help="Path to the image folder containing clean and adversarial images.")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs to train the model.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--lambda_perceptual", type=float, default=5.0,
                        help="Weight for perceptual loss.")
    parser.add_argument("--lambda_l1", type=float, default=1.0,
                        help="Weight for L1 loss.")
    parser.add_argument("--ssim", type=float, default=10.0,
                        help="Weight for SSIM loss")
    parser.add_argument("--checkpoint_path", type=str, default='',
                        help="Path to save the trained model and loss figures.")
    return parser.parse_args()

class DenoisingDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.clean_images = []
        self.adversarial_images = []

        for filename in os.listdir(image_folder):
            if filename.endswith("_clean.jpg"):
                clean_image_path = os.path.join(image_folder, filename)
                adversarial_image_path = clean_image_path.replace("_clean.jpg", "_adv.jpg")
                if os.path.exists(adversarial_image_path):
                    self.clean_images.append(clean_image_path)
                    self.adversarial_images.append(adversarial_image_path)

        print(f"Data loaded: {len(self.clean_images)} clean images and {len(self.adversarial_images)} adversarial images.")

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        clean_img = Image.open(self.clean_images[idx]).convert("RGB")
        adversarial_img = Image.open(self.adversarial_images[idx]).convert("RGB")

        if self.transform:
            clean_img = self.transform(clean_img)
            adversarial_img = self.transform(adversarial_img)

        return clean_img, adversarial_img


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)), 
])

def get_ddpm_model():
    model = UNet2DModel.from_pretrained(
        "/path_to_model",
        weight_dtype=torch.float32
    )
    model.to(device)
    scheduler = DDPMScheduler.from_pretrained("")  
    return model, scheduler

def perceptual_loss_fn(pred, target):

    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    resnet.eval()
    pred_features = extract_features(pred, resnet)
    target_features = extract_features(target, resnet)
    loss = F.mse_loss(pred_features, target_features)
    return loss

def extract_features(image, model):
    layers = list(model.children())[:-1]  
    model = torch.nn.Sequential(*layers)
    with torch.no_grad():
        features = model(image)
    features = features.view(features.size(0), -1)
    return features

def train_model(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(args.checkpoint_path, exist_ok=True)

    dataset = DenoisingDataset(args.image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"Starting training with {len(dataset)} samples...")

    model, scheduler = get_ddpm_model()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    epoch_mse_losses = []
    epoch_perceptual_losses = []
    epoch_ssim_losses = []
    epoch_total_losses = []

    for epoch in range(args.epochs):
        model.train()

        running_mse_loss = 0.0
        running_perceptual_loss = 0.0
        running_ssim_loss = 0.0
        running_total_loss = 0.0

        epoch_start_time = time.time() 
        total_batches = len(dataloader)

        print(f"Epoch {epoch+1}/{args.epochs} started...")

        for i, (clean_img, adversarial_img) in enumerate(dataloader):
            batch_start_time = time.time()  
            clean_img = clean_img.to(device)
            adversarial_img = adversarial_img.to(device)

            timesteps = torch.randint(
                low=0,
                high=scheduler.config.num_train_timesteps,
                size=(clean_img.size(0),),
                dtype=torch.long,
                device=clean_img.device
            )

            denoised_img = model(adversarial_img, timestep=timesteps)
            denoised_img = denoised_img.sample  

            mse_loss_val = F.mse_loss(denoised_img, clean_img)
            perceptual_loss_val = perceptual_loss_fn(denoised_img, clean_img)
            ssim_loss_val = 1 - ssim(
                denoised_img.clamp(-1, 1),
                clean_img.clamp(-1, 1),
                data_range=2.0
            )

            total_loss_val = (
                mse_loss_val
                + args.lambda_perceptual * perceptual_loss_val
                + args.ssim * ssim_loss_val
            )

            optimizer.zero_grad()
            total_loss_val.backward()
            optimizer.step()

            running_mse_loss += mse_loss_val.item()
            running_perceptual_loss += perceptual_loss_val.item()
            running_ssim_loss += ssim_loss_val.item()
            running_total_loss += total_loss_val.item()

            if (i + 1) % 10 == 0:
                batch_time = time.time() - batch_start_time 
                it_per_sec = 1.0 / batch_time 
                batches_left = total_batches - i - 1
                time_left = batches_left / it_per_sec
                remaining_time = str(time.strftime("%H:%M:%S", time.gmtime(time_left))) 

                print(f"[epoch:{epoch+1}, Batch:{i+1}/{total_batches}, "
                      f"mse_loss={mse_loss_val.item():.6f}, "
                      f"pct_loss={perceptual_loss_val.item():.6f}, "
                      f"ssim_loss={ssim_loss_val.item():.6f}, "
                      f"total_loss={total_loss_val.item():.6f}, "
                      f"{str(time.strftime('%H:%M:%S', time.gmtime(time.time() - epoch_start_time)))}<"
                      f"{remaining_time}, {it_per_sec:.2f}it/s]")

        avg_mse_loss = running_mse_loss / total_batches
        avg_perceptual_loss = running_perceptual_loss / total_batches
        avg_ssim_loss = running_ssim_loss / total_batches
        avg_total_loss = running_total_loss / total_batches

        epoch_mse_losses.append(avg_mse_loss)
        epoch_perceptual_losses.append(avg_perceptual_loss)
        epoch_ssim_losses.append(avg_ssim_loss)
        epoch_total_losses.append(avg_total_loss)

        print(f"Epoch {epoch+1} completed. "
              f"Avg MSE: {avg_mse_loss:.6f}, "
              f"Avg Perceptual: {avg_perceptual_loss:.6f}, "
              f"Avg SSIM: {avg_ssim_loss:.6f}, "
              f"Avg Total: {avg_total_loss:.6f}")

        # checkpoint_path = os.path.join(args.checkpoint_path, f"ddpm_model_{epoch+1}.pth")
        # torch.save(model.state_dict(), checkpoint_path)
        # print(f"Model checkpoint saved to {checkpoint_path}.")

    epochs_range = range(1, args.epochs + 1)

    # 1. MSE Loss
    plt.figure()
    plt.plot(epochs_range, epoch_mse_losses, marker='o')
    plt.title("MSE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(os.path.join(args.checkpoint_path, "mse_loss.png"))
    plt.close()

    # 2. Perceptual Loss
    plt.figure()
    plt.plot(epochs_range, epoch_perceptual_losses, marker='o')
    plt.title("Perceptual Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Perceptual")
    plt.tight_layout()
    plt.savefig(os.path.join(args.checkpoint_path, "perceptual_loss.png"))
    plt.close()

    # 3. SSIM Loss
    plt.figure()
    plt.plot(epochs_range, epoch_ssim_losses, marker='o')
    plt.title("SSIM Loss (1 - SSIM)")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(args.checkpoint_path, "ssim_loss.png"))
    plt.close()

    # 4. Total Loss
    plt.figure()
    plt.plot(epochs_range, epoch_total_losses, marker='o')
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Total")
    plt.tight_layout()
    plt.savefig(os.path.join(args.checkpoint_path, "total_loss.png"))
    plt.close()

    print("All loss curves have been saved separately.")

if __name__ == "__main__":

    args = parse_args()

    train_model(args)
