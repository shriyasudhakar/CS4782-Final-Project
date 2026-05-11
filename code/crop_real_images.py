from torchvision import transforms
from PIL import Image
from pathlib import Path
import os
import torchvision.utils as vutils

IN_DIR = Path("./coco_val_images")
OUT_DIR = Path("./real_coco_processed")
os.makedirs(OUT_DIR, exist_ok=True)

path_to_img = []
for f in os.listdir(IN_DIR):
    if f.endswith('png') or f.endswith('jpg'):
        path_to_img.append(os.path.join(IN_DIR, f))

test_transform = transforms.Compose([
    transforms.CenterCrop(360),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

for idx in range(len(path_to_img)):
    image = Image.open(path_to_img[idx]).convert('RGB')
    image = test_transform(image)
    vutils.save_image(image, OUT_DIR / F"real_image_{idx}.png")