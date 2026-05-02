from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as numpy
import os
from PIL import Image

# Section 3.3: "we apply standard data augmentation procedures including horizontal flipping and random cropping to cover image C in our pre-processing pipeline"

class Div2KDataset(Dataset):
    def __init__(self, root, transforms=None):
        # Sort filenames so iteration order is deterministic across runs / OSes.
        # os.listdir() returns files in arbitrary order otherwise.
        self.path_to_img = [
            os.path.join(root, f)
            for f in sorted(os.listdir(root))
            if f.endswith('png')
        ]

        self.transforms = transforms

    def __len__(self):
        return len(self.path_to_img)
    
    def __getitem__(self, idx):
        image = Image.open(self.path_to_img[idx]).convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        return image