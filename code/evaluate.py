import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure

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

def rs_bpp(D, message, decoded_message):
  """
  Inputs:
  - D: how many bits you store per pixel
  - message: the original random message N x D x H x W (binary 0s and 1s)
  - decoded_message: raw decoder output N x D x H x W (floats/logits)

  Returns: scalar RS-BPP value
  """
  # decoder outputs raw floats, so threshold to get predicted bits
  predicted_bits = (decoded_message > 0).float()

  # compare every entry in D x H x W across the batch
  # wrong = 1 where they differ, 0 where they match
  wrong = (predicted_bits != message).float()

  # p = total wrong bits / total bits (single scalar across entire batch)
  p = wrong.mean().item()

  # RS-BPP formula: clamp to 0 if decoder is worse than random (p > 0.5)
  return max(0.0, D * (1 - 2 * p))

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
  metric = StructuralSimilarityIndexMeasure(data_range=2.0)
  return metric(stego_image, cover_image).item()

def evaluate(encoder, decoder, dataloader, D, device="cpu"):
  """
  Inputs:
  - encoder: trained encoder model
  - decoder: trained decoder model
  - dataloader: DataLoader over the validation set
  - D: bits per pixel (must match what encoder/decoder were trained with)
  - device: "cpu" or "cuda"

  Returns: dict with average RS-BPP, PSNR, SSIM across the validation set
  """
  
  #Switch the models from training mode to evaluation mode.
  #BatchNorm behaves differently during evaluation (uses running mean and variance)
  encoder.eval()
  decoder.eval()

  total_rs_bpp = 0.0
  total_psnr = 0.0
  total_ssim = 0.0
  num_batches = 0

  with torch.no_grad():
    for cover_image in dataloader: #cover_image is shape (N, 3, H, W)
      cover_image = cover_image.to(device)
      N, _, H, W = cover_image.shape

      # generate random binary message for this batch
      message = torch.randint(0, 2, (N, D, H, W), dtype=torch.float, device=device)

      # encode and decode
      stego_image = encoder(cover_image, message)
      decoded_message = decoder(stego_image)

      # accumulate metrics
      total_rs_bpp += rs_bpp(D, message, decoded_message)
      total_psnr += psnr(cover_image, stego_image)
      total_ssim += ssim(cover_image, stego_image)
      num_batches += 1

  return {
    "RS-BPP": total_rs_bpp / num_batches,
    "PSNR": total_psnr / num_batches,
    "SSIM": total_ssim / num_batches,
  }