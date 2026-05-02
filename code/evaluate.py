import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm

"""
Evaluation functions for the encoder+decoder.
- Loads checkpoints from training
- Calculates metrics on validation set
- Prints metrics

"""

#Metric 1: RS-BPP (Reed-Solomon Bits Per Pixel)
#"How much real, recoverable information did we actually stuff into each pixel?"

# For each test image:
# Generate a random message, encode it, then decode it
# Compute the bit error rate p (fraction of bits the decoder gets wrong)
# RS-BPP = D × (1 - 2p), where D is your data depth (bits per pixel attempted)
# This tells you: given Reed-Solomon error correction,
# how many reliable bits per pixel can you actually transmit?
# If your decoder has 10% error rate at D=6, then RS-BPP = 6 × (1 - 0.2) = 4.8

#Metric 2: PSNR (Peak Signal-to-Noise Ratio)
# Compares cover image to stego image pixel-by-pixel. 20*log10(max_pixel) - 10*log10(MSE)
# Higher means better

#Metric 3: SSIM (Structural Similarity Index)
# Measures structural similarity between cover and stego images
# accounting for luminance, contrast, and structure
#Values in [-1, 1], closer to 1 is better.

def psnr(cover_image, stego_image):
  """
  Inputs:
  - cover_image: N x 3 x H x W (normalized to [-1, 1])
  - stego_image: N x 3 x H x W (normalized to [-1, 1])

  Returns: scalar average PSNR in dB across the batch
  """
  # pixel range is 2.0 for images in [-1, 1]
  max_val = 2.0
  mse = F.mse_loss(stego_image, cover_image)
  return (10 * torch.log10(max_val ** 2 / mse)).item()

def ssim(cover_image, stego_image):
  """
  Inputs:
  - cover_image: N x 3 x H x W (normalized to [-1, 1])
  - stego_image: N x 3 x H x W (normalized to [-1, 1])

  Returns: scalar SSIM value, closer to 1 is better
  """
  device = cover_image.device
  metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
  return metric(stego_image, cover_image).item()

def evaluate(encoder, decoder, dataloader, D, device="cpu"):
  """
  Inputs:
  - encoder: trained encoder model
  - decoder: trained decoder model
  - dataloader: DataLoader over the validation set
  - D: bits per pixel (must match what encoder/decoder were trained with)
  - device: "cpu" or "cuda"

  Returns: dict with RS-BPP (aggregated over the whole val set) and
           image-weighted average PSNR / SSIM.

  Notes:
  - RS-BPP is computed by summing wrong-bit counts across the entire val set
    and applying the formula once. Per-batch averaging would inflate it
    because of the max(0, ·) clamp (non-linear → mean(max(...)) ≠ max(mean(...))).
  - PSNR / SSIM are weighted by batch size N so the smaller final batch
    (e.g. 4 images out of 100) doesn't get over-weighted.
  """

  #Switch the models from training mode to evaluation mode.
  #BatchNorm behaves differently during evaluation (uses running mean and variance)
  encoder.eval()
  decoder.eval()

  total_wrong = 0
  total_bits  = 0
  psnr_sum = 0.0
  ssim_sum = 0.0
  total_images = 0

  with torch.no_grad():
    for cover_image in tqdm(dataloader, desc="Evaluating"): #cover_image is shape (N, 3, H, W)
      cover_image = cover_image.to(device)
      N, _, H, W = cover_image.shape

      # generate random binary message for this batch
      message = torch.randint(0, 2, (N, D, H, W), dtype=torch.float, device=device)

      # encode and decode
      stego_image = encoder(cover_image, message)
      decoded_message = decoder(stego_image)

      # accumulate raw bit-error counts so we can apply the RS-BPP formula
      # ONCE at the end (avoids the max(0, ·) clamp bias from per-batch averaging)
      predicted_bits = (decoded_message > 0).float()
      total_wrong += (predicted_bits != message).sum().item()
      total_bits  += message.numel()

      # weight image-level metrics by batch size so the trailing partial batch
      # doesn't get over-weighted
      psnr_sum += psnr(cover_image, stego_image) * N
      ssim_sum += ssim(cover_image, stego_image) * N
      total_images += N

  p = total_wrong / total_bits
  rs_bpp = max(0.0, D * (1 - 2 * p))
  acc = 1.0 - p

  return {
    "RS-BPP": rs_bpp,
    "PSNR": psnr_sum / total_images,
    "SSIM": ssim_sum / total_images,
    "Acc": acc,
  }
