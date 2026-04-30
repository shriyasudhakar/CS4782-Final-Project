from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as numpy
import os
from PIL import Image

# Section 3.3: "we apply standard data augmentation procedures including horizontal flipping and random cropping to cover image C in our pre-processing pipeline"

class Div2KDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.path_to_img = []
        for f in os.listdir(root):
            if f.endswith('png'):
                self.path_to_img.append(os.path.join(root, f))

        self.transforms = transforms

    def __len__(self):
        return len(self.path_to_img)
    
    def __getitem__(self, idx):
        image = Image.open(self.path_to_img[idx]).convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        return image