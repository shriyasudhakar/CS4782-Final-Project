import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import argparse

from critic import Critic
from decoder import Decoder
from encoders import ResidualEncoder
from evaluate import evaluate



class Trainer:
    """
    This training class jointly optimizes three networks:
    - Encoder: hides a binary message inside an image
    - Decoder: recovers the hidden message
    - Critic: distinguishes real vs generated images (Wasserstein GAN)

    For each batch:
    1. Sample a random binary message M ~ Ber(0.5)
    2. Update the critic using Wasserstein loss:
    Lc = C(real) - C(fake)
    3. Update encoder + decoder using:
    L = Ld + Ls + Lr
    where:
        Ld: decoding loss (binary cross entropy)
        Ls: similarity loss (MSE between cover and stego image)
        Lr: realism loss from critic

    After each epoch, evaluate using:
        RS-BPP (message capacity)
        PSNR (pixel-level distortion)
        SSIM (perceptual similarity)
    """
    def __init__(self, encoder, decoder, critic, D, device):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.critic = critic.to(device)

        self.device = device
        self.D = D

        self.enc_dec_opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=1e-4,
        )
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

        self.grad_clip = 0.25
        self.critic_clip = 0.1

    def sample_message(self, N, H, W):
        """
        Inputs:
        - N: batch size
        - D: bits per pixel (data depth)
        - H, W: spatial dimensions
        - device: torch device (cpu or cuda)

        Returns:
        - message: N x D x H x W tensor of binary values {0,1}

        Description:
        - Generates a random binary message for each image in the batch
        - Each pixel stores D bits
        - Values are sampled from a Bernoulli(0.5) distribution
        """
        return torch.randint(0, 2, (N, self.D, H, W), device=self.device).float()

    def similarity_loss(self, cover, generated):
        """
        Inputs:
        - cover: N x 3 x H x W original image
        - generated: N x 3 x H x W stego image

        Returns:
        - scalar similarity loss

        Description:
        - Computes normalized mean squared error between cover and stego images
        - Matches paper formulation:
            Ls = (1 / (3 * H * W)) * ||cover - generated||^2
        - Encourages minimal visual distortion
        """
        _, _, H, W = cover.shape
        return ((cover - generated) ** 2).sum(dim=(1, 2, 3)).mean() / (3 * H * W)

    def train_epoch(self, loader):

        self.encoder.train()
        self.decoder.train()
        self.critic.train()

        total = {"Lc": 0.0, "Ld": 0.0, "Ls": 0.0, "Lr": 0.0, "Acc": 0.0}
        steps = 0

        for cover in loader:
            cover = cover.to(self.device)
            N, _, H, W = cover.shape
            
            # sample message to hide for this batch
            M = self.sample_message(N, H, W)

            # Critic 
            with torch.no_grad():
                fake = self.encoder(cover, M)

            # Critic Update (Wasserstein GAN + Gradient clipping (stability) + Weight clipping to enforce Lipschitz constraint)
            real_score = self.critic(cover).mean()
            fake_score = self.critic(fake).mean()
            Lc = real_score - fake_score

            self.critic_opt.zero_grad()
            Lc.backward()
            clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            self.critic_opt.step()

            for p in self.critic.parameters():
                p.data.clamp_(-self.critic_clip, self.critic_clip)

            # Encoder-Decoder Loss & Updates
            fake = self.encoder(cover, M)
            decoded = self.decoder(fake)

            # Ld (decoding loss): Binary cross entropy between decoded message and original message
            Ld = F.binary_cross_entropy_with_logits(decoded, M)
            # Ls (similarity loss): MSE between cover and stego image
            Ls = self.similarity_loss(cover, fake)
            # Lr (realness loss): Critic score of generated image
            Lr = self.critic(fake).mean()

            loss = Ld + 100.0 * Ls + Lr

            self.enc_dec_opt.zero_grad()
            loss.backward()
            clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                self.grad_clip,
            )
            self.enc_dec_opt.step()

            with torch.no_grad():
                acc = ((decoded >= 0) == (M >= 0.5)).float().mean() # Measures fraction of correctly recovered bits

            total["Lc"] += Lc.item()
            total["Ld"] += Ld.item()
            total["Ls"] += Ls.item()
            total["Lr"] += Lr.item()
            total["Acc"] += acc.item()
            steps += 1

        return {k: v / steps for k, v in total.items()}

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}")

            train_metrics = self.train_epoch(train_loader)

            print(
                f"Lc: {train_metrics['Lc']:.4f} | "
                f"Ld: {train_metrics['Ld']:.4f} | "
                f"Ls: {train_metrics['Ls']:.6f} | "
                f"Lr: {train_metrics['Lr']:.4f} | "
                f"Acc: {train_metrics['Acc']:.4f}"
            )

            # Evaluation metrics, see evaluate.py for details
            eval_metrics = evaluate(self.encoder, self.decoder, val_loader, self.D, device=self.device)

            print(
                f"RS-BPP: {eval_metrics['RS-BPP']:.4f} | "
                f"PSNR: {eval_metrics['PSNR']:.2f} | "
                f"SSIM: {eval_metrics['SSIM']:.4f}"
            )

class FakeLoader:
    """
    Generates batches of synthetic images instead of loading from disk.
    """
    def __init__(self, batch_size, num_batches, H, W):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.H = H
        self.W = W

    def __iter__(self):
        for _ in range(self.num_batches):
            yield torch.rand(self.batch_size, 3, self.H, self.W) * 2 - 1

    def __len__(self):
        return self.num_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--data-depth", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = ResidualEncoder(args.data_depth)
    decoder = Decoder(args.data_depth)
    critic = Critic()

    trainer = Trainer(encoder, decoder, critic, args.data_depth, device)

    train_loader = FakeLoader(args.batch_size, 10, args.crop_size, args.crop_size)
    val_loader = FakeLoader(args.batch_size, 5, args.crop_size, args.crop_size)

    trainer.train(train_loader, val_loader, args.epochs)


if __name__ == "__main__":
    main()

# python3 train.py --epochs 1 --batch-size 2 --crop-size 64 --data-depth 2