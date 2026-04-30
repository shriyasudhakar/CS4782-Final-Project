import torch
from decoder import Decoder
from encoders import DenseEncoder
from datasetloader import DataLoader, Div2KDataset
import urllib.request
from torchvision import transforms
from pathlib import Path
import zipfile

def _denorm(x):
    # undo image normalization so we can visualize them during wandb logging
    return (x * 0.5 + 0.5).clamp(0, 1)

def _qualitative_samples_dict(encoder, decoder, fixed_batch, D, device, max_images=4):
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
        cover = fixed_batch.to(device)[:max_images]
        N, _, H, W = cover.shape
        # use a fixed seed to generate the message so that it's the same every epoch for this panel
        g = torch.Generator(device=device).manual_seed(0)
        M = torch.randint(0, 2, (N, D, H, W), generator=g, device=device).float()

        stego = encoder(cover, M)

        cover_v = _denorm(cover)
        stego_v = _denorm(stego)
        print(type(cover_v))
        print(type(stego_v))
        print(cover_v.shape)
        # need to save stego file

validation_transform = transforms.Compose([
    transforms.CenterCrop(360), # keep crop deterministic
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # same normalization as train
])

url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
output_path = Path("../data/div2k_valid.zip")

# Downloads the file and saves it to the specified path
urllib.request.urlretrieve(url, output_path)

with zipfile.ZipFile(output_path, 'r') as zip_ref:
    zip_ref.extractall("div2k_valid")

root_path_to_image_valid = '../data/div2k_valid'

# set the batch size and workers here
batch_size = 100 #??
num_workers = 0

valid_dataset = Div2KDataset(root_path_to_image_valid, transforms=validation_transform)
val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# get all images???
val_images = next(iter(val_loader))

# num of bits to hide in pixel
D = 1
# load model from checkpoint
encoder_model = DenseEncoder(D)
decoder_model = Decoder(D)
# assuming running from /code
PATH= "../checkpoints/residual_D1.pt"
checkpoint = torch.load(PATH, weights_only=True)
encoder_model.load_state_dict(checkpoint['encoder_state_dict'])
decoder_model.load_state_dict(checkpoint['decoder_state_dict'])
device = encoder_model.dev
        
