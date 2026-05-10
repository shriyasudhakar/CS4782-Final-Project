import torch
from decoder import Decoder
from encoders import DenseEncoder, ResidualEncoder, BasicEncoder
from datasetloader import DataLoader, Div2KDataset
import urllib.request
from torchvision import transforms
import torchvision.utils as vutils
from pathlib import Path
import zipfile
import os

def _denorm(x):
    # undo image normalization so we can visualize them during wandb logging
    return (x * 0.5 + 0.5).clamp(0, 1)

def _qualitative_samples_dict(encoder, decoder, fixed_batch, D, device, encoder_type="Dense", save_dir=f"../data/DIV2K_valid_HR_outputs"):
    """
    Saves cover images and generated encoded images to evaluate on StegExpose.

    Panels:
      - cover:     original images
      - stego:     encoder output given a random message
      - residual:  visual differences between original and encoded image. bright spots show differences
    """

    #switch models to eval mode
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        cover = fixed_batch.to(device)
        N, _, H, W = cover.shape
        # use a fixed seed to generate the message so that it's the same every epoch for this panel
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
        os.makedirs(output_dir / f"{encoder_type}_D={D}" / "stego", exist_ok=True)
        os.makedirs(output_dir / f"{encoder_type}_D={D}" / "residual", exist_ok=True)

        # Loop through the batch and save each image
        for i in range(cover_v.size(0)):
            vutils.save_image(stego_v[i], output_dir / f"{encoder_type}_D={D}" / "stego" / f"stego_image_{i}.png")
            vutils.save_image(residual_v[i], output_dir / f"{encoder_type}_D={D}" / "residual" / f"residual_image_{i}.png")

def _load_validation_data(output_path):
    url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
    # Downloads the file and saves it to the specified path
    urllib.request.urlretrieve(url, output_path)

    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall("../data")

output_path = Path("../data/div2k_valid.zip")
root_path_to_image_valid = '../data/DIV2K_valid_HR'
if os.path.exists(root_path_to_image_valid):
    print("Validation data found. Skipping download..")
else:
    _load_validation_data(output_path)

# set the batch size and workers here
batch_size = 100 #??
num_workers = 0

validation_transform = transforms.Compose([
    transforms.CenterCrop(360), # keep crop deterministic
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # same normalization as train
])
valid_dataset = Div2KDataset(root_path_to_image_valid, transforms=validation_transform)
val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

val_images = []
# get all images???
for images in val_loader:
    print(images.shape)
    val_images: torch.Tensor = images
    # 'images' is a batch of tensors; 'labels' are the class indices

# num of bits to hide in pixel
D = 6
# load model from checkpoint
encoder_model = BasicEncoder(D)
decoder_model = Decoder(D)
# assuming running from /code and checkpoint is downloaded
PATH= "../checkpoints/basic_D6.pt"
device = val_images.device
print(f"Current device is {device}")
checkpoint = torch.load(PATH, weights_only=True, map_location=device)
encoder_model.load_state_dict(checkpoint['encoder_state_dict'])
decoder_model.load_state_dict(checkpoint['decoder_state_dict'])

_qualitative_samples_dict(encoder_model, decoder_model, val_images, D, device, encoder_type="Basic", save_dir="../data/DIV2K_valid_HR_outputs")
        
