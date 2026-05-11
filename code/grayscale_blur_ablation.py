import torch
from decoder import Decoder
from encoders import BasicEncoder, ResidualEncoder, DenseEncoder
from datasetloader import DataLoader, Div2KDataset
from evaluate import evaluate
import urllib.request
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import csv
import zipfile
import os

"""
This script evaluates how our trained model performs on grayscale and blurred versions of
the validation images from the Div2K set. Our model was trained on the training set of 
Div2k without grayscale/blurring. For this ablation, I only use the basic encoder model variant
with messages of depth 6.

First, it creates two folders with grayscale and blurred variants of the validation set: 
- DIV2K_valid_grayscale
- DIV2K_valid_blurred

Then, it passes each set through the trained model with randomized messages.
And evaluates RS-BPP, PSNR, and SSIM on it using evaluate.py

And saves these folders:
- DIV2K_valid_grayscale_outputs
  - eval_metrics.csv (containing quantitative results from evalute.py)
  - residual (containing a visual difference between original and encoded images)
  - stego (containing encoded images)
- DIV2K_valid_blurred_outputs
  - eval_metrics.csv 
  - residual
  - stego

"""

# Architecture: one of "basic", "residual", "dense"
ARCH = "basic"
# Bits hidden per pixel (must match the checkpoint).
D = 6
# Checkpoint path: must be consistent with ARCH and D.
CKPT_PATH = "../checkpoints/basic_D6.pt"

# Gaussian blur strength applied
# Mild: (7, 2.0) | Moderate: (15, 4.0) | Strong: (25, 8.0) | Extreme: (51, 16.0)
BLUR_KERNEL_SIZE = 15
BLUR_SIGMA = 4.0
#sigma controls the weighted averaging within the kernel size
#small sigma means most of the weight is peaked at the center. barely any blurring
#large sigma means the weights are more spread out across the kernel

BATCH_SIZE = 16
NUM_WORKERS = 0

ENCODER_CLASSES = {
    "basic":    BasicEncoder,
    "residual": ResidualEncoder,
    "dense":    DenseEncoder,
}


def _denorm(x):
    # undo image normalization so we can visualize them during wandb logging
    return (x * 0.5 + 0.5).clamp(0, 1)

def _qualitative_samples_dict(encoder, decoder, fixed_batch, D, device,
                              save_dir="../data/DIV2K_valid_HR_outputs",
                              source_filenames=None):
    """
    Saves cover images and generated encoded images to evaluate on StegExpose.

    Panels:
      - cover:     original images
      - stego:     encoder output given a random message
      - residual:  visual differences between original and encoded image. bright spots show differences

    If source_filenames is provided, output files are named after the source
    image (e.g. stego_0801.png) so they can be matched back to the input set.
    Otherwise we fall back to stego_image_<i>.png.
    """

    #switch models to eval mode
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        cover = fixed_batch.to(device)
        N, _, H, W = cover.shape
        # use a fixed seed to generate the message so that it's the same every epoch for this panel.
        # MPS doesn't support torch.Generator(device="mps"), so seed on CPU and move.
        if device.type == "mps":
            g = torch.Generator().manual_seed(0)
            M = torch.randint(0, 2, (N, D, H, W), generator=g).float().to(device)
        else:
            g = torch.Generator(device=device).manual_seed(0)
            M = torch.randint(0, 2, (N, D, H, W), generator=g, device=device).float()

        stego = encoder(cover, M)

        cover_v = _denorm(cover)
        stego_v = _denorm(stego)
        #The residual image will show bright spots where the encoder modified the cover image
        #Helps see if the embedding is spatially uniform, and if it hides bits in edgy textured areas or smooth areas
        residual_v = (stego_v - cover_v).abs().mul(10).clamp(0, 1)
        output_dir = Path(save_dir)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "stego", exist_ok=True)
        os.makedirs(output_dir / "residual", exist_ok=True)

        # Loop through the batch and save each image
        for i in range(cover_v.size(0)):
            if source_filenames is not None:
                stem = Path(source_filenames[i]).stem  # "0801" from "0801.png"
                stego_name    = f"stego_{stem}.png"
                residual_name = f"residual_{stem}.png"
            else:
                stego_name    = f"stego_image_{i}.png"
                residual_name = f"residual_image_{i}.png"
            vutils.save_image(stego_v[i],    output_dir / "stego"    / stego_name)
            vutils.save_image(residual_v[i], output_dir / "residual" / residual_name)


def _build_perturbed_dataset(src_dir, dst_dir, pil_transform):
    """
    Read each PNG in src_dir, apply pil_transform (PIL -> PIL), save result
    as PNG in dst_dir. Skips if dst_dir already has the same number of PNGs.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    src_files = sorted([f for f in os.listdir(src_dir) if f.endswith("png")])

    if dst_dir.exists():
        existing = [f for f in os.listdir(dst_dir) if f.endswith("png")]
        if len(existing) == len(src_files):
            print(f"Perturbed set found at {dst_dir}, skipping rebuild.")
            return
    os.makedirs(dst_dir, exist_ok=True)

    for fname in tqdm(src_files, desc=f"Building {dst_dir.name}"):
        img = Image.open(src_dir / fname).convert("RGB")
        img = pil_transform(img)
        img.save(dst_dir / fname)
    print(f"Built perturbed set at {dst_dir} ({len(src_files)} images).")


def _run_ablation(name, dataset_root, encoder, decoder, D, arch, device,
                  batch_size, num_workers, validation_transform):
    """
    Run encoder/decoder over a (perturbed) validation set:
      - dump qualitative stego + residual images to
        ../data/{name}_outputs/D=<D>_model=<arch>/
      - compute RS-BPP / PSNR / SSIM via evaluate() and write eval_metrics.csv
        into the same nested dir
    """
    out_dir = Path(f"../data/{name}_outputs") / f"D={D}_model={arch}"
    os.makedirs(out_dir, exist_ok=True)

    dataset = Div2KDataset(dataset_root, transforms=validation_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # qualitative panels: pull the first batch and save stego/residual images.
    # Since shuffle=False and dataset paths are sorted, the first batch always
    # contains the first `batch_size` files from the sorted listing, so we can
    # name each output after its source filename.
    first_batch = next(iter(loader))
    first_batch_filenames = [
        os.path.basename(p) for p in dataset.path_to_img[:first_batch.size(0)]
    ]
    _qualitative_samples_dict(encoder, decoder, first_batch, D, device,
                              save_dir=str(out_dir),
                              source_filenames=first_batch_filenames)

    # quantitative metrics over the full perturbed val set
    metrics = evaluate(encoder, decoder, loader, D, device=device)
    print(f"[{name}] {metrics}")

    with open(out_dir / "eval_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])


def _load_validation_data(output_path):
    url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
    # Downloads the file and saves it to the specified path
    urllib.request.urlretrieve(url, output_path)

    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall("../data")

def main():
    output_path = Path("../data/div2k_valid.zip")
    root_path_to_image_valid = '../data/DIV2K_valid_HR'
    if os.path.exists(root_path_to_image_valid):
        print("Validation data found. Skipping download..")
    else:
        _load_validation_data(output_path)

    # Build grayscale and blurred copies of the val set on disk
    grayscale_root = '../data/DIV2K_valid_grayscale'
    blurred_root   = '../data/DIV2K_valid_blurred'

    # num_output_channels=3 keeps it as a 3-channel RGB tensor (R=G=B), which the
    # encoder requires since its first conv expects 3 input channels.
    _build_perturbed_dataset(
        root_path_to_image_valid, grayscale_root,
        transforms.Grayscale(num_output_channels=3),
    )
    _build_perturbed_dataset(
        root_path_to_image_valid, blurred_root,
        transforms.GaussianBlur(kernel_size=BLUR_KERNEL_SIZE, sigma=BLUR_SIGMA),
    )

    validation_transform = transforms.Compose([
        transforms.CenterCrop(360), # keep crop deterministic
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # same normalization as train
    ])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Current device is {device}")

    # load encoder + decoder from the configured checkpoint
    EncoderClass = ENCODER_CLASSES[ARCH]
    encoder_model = EncoderClass(D).to(device)
    decoder_model = Decoder(D).to(device)
    checkpoint = torch.load(CKPT_PATH, weights_only=True, map_location=device)
    encoder_model.load_state_dict(checkpoint['encoder_state_dict'])
    decoder_model.load_state_dict(checkpoint['decoder_state_dict'])

    # Run the two ablations
    _run_ablation("DIV2K_valid_grayscale", grayscale_root,
                  encoder_model, decoder_model, D, ARCH, device,
                  BATCH_SIZE, NUM_WORKERS, validation_transform)

    _run_ablation("DIV2K_valid_blurred", blurred_root,
                  encoder_model, decoder_model, D, ARCH, device,
                  BATCH_SIZE, NUM_WORKERS, validation_transform)


if __name__ == "__main__":
    main()

